"""
EGI DataHub Zarr Toolkit
========================

A Python toolkit for reading and writing Zarr v3 stores directly from/to 
EGI DataHub using the Onedata REST API.

Requirements:
    - requests
    - zarr >= 3.0
    - xarray

Usage:
    from egi_datahub_zarr import DataHubClient
    
    # Initialize client
    client = DataHubClient(token="your-token")
    
    # List spaces
    spaces = client.list_spaces()
    
    # Open Zarr store for reading
    ds = client.open_zarr("Reliance/FAIR2Adapt/CS1/sample_climate_data.zarr")
    
    # Write Zarr store
    client.to_zarr(ds, "Reliance/FAIR2Adapt/CS1/my_new_data.zarr")

Author: Science Live
License: CC-BY-SA 4.0
"""

import os
import requests
from typing import Optional, Dict, List, Any
from collections.abc import Iterable

# Zarr imports
import zarr
from zarr.abc.store import Store
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype

# Optional xarray import
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_ONEZONE = "datahub.egi.eu"
DEFAULT_PROVIDER = "cesnet-oneprovider-01.datahub.egi.eu"


# =============================================================================
# OnedataZarrStore - Zarr v3 Compatible Store
# =============================================================================

class OnedataZarrStore(Store):
    """
    Zarr v3 store implementation for EGI DataHub via Onedata REST API.
    
    This store allows direct reading and writing of Zarr stores on EGI DataHub
    without downloading the entire dataset locally.
    
    Parameters
    ----------
    root_file_id : str
        The Onedata file ID of the Zarr store root directory
    token : str
        Onedata access token
    provider : str, optional
        Oneprovider hostname (default: cesnet-oneprovider-01.datahub.egi.eu)
    read_only : bool, optional
        If True, disable write operations (default: True)
    
    Examples
    --------
    >>> store = OnedataZarrStore(zarr_id, token, read_only=True)
    >>> ds = xr.open_zarr(store, consolidated=False, zarr_format=3)
    """
    
    def __init__(
        self, 
        root_file_id: str, 
        token: str, 
        provider: str = DEFAULT_PROVIDER,
        read_only: bool = True
    ):
        self.root_id = root_file_id
        self.token = token
        self.provider = provider
        self.headers = {"X-Auth-Token": token}
        self._file_cache: Dict[str, str] = {}  # path -> file_id
        self._read_only = read_only
        self._is_open = True
        
    def _api_url(self, endpoint: str) -> str:
        """Build API URL for Oneprovider endpoint."""
        return f"https://{self.provider}/api/v3/oneprovider/{endpoint}"
    
    def _list_dir(self, dir_id: str) -> List[Dict[str, Any]]:
        """List directory contents."""
        all_children = []
        url = self._api_url(f"data/{dir_id}/children")
        
        while url:
            resp = requests.get(url, headers=self.headers)
            if not resp.ok:
                return []
            data = resp.json()
            all_children.extend(data.get('children', []))
            
            if data.get('isLast', True):
                break
            token = data.get('nextPageToken')
            if token:
                url = self._api_url(f"data/{dir_id}/children?page_token={token}")
            else:
                break
                
        return all_children
    
    def _resolve_path(self, path: str) -> Optional[str]:
        """
        Resolve a path relative to the store root to a file ID.
        
        Returns None if the path does not exist.
        """
        path = path.strip('/')
        if not path:
            return self.root_id
        
        if path in self._file_cache:
            return self._file_cache[path]
        
        parts = path.split('/')
        current_id = self.root_id
        current_path = ""
        
        for part in parts:
            children = self._list_dir(current_id)
            child = next((c for c in children if c.get('name') == part), None)
            if not child:
                return None
            current_id = child['file_id']
            current_path = f"{current_path}/{part}" if current_path else part
            self._file_cache[current_path] = current_id
        
        return current_id
    
    def _ensure_parent_dirs(self, path: str) -> str:
        """
        Create parent directories if they don't exist.
        
        Returns the file ID of the immediate parent directory.
        """
        path = path.strip('/')
        parts = path.split('/')
        
        if len(parts) == 1:
            return self.root_id
        
        current_id = self.root_id
        current_path = ""
        
        for part in parts[:-1]:
            current_path = f"{current_path}/{part}" if current_path else part
            
            if current_path in self._file_cache:
                current_id = self._file_cache[current_path]
                continue
            
            children = self._list_dir(current_id)
            child = next((c for c in children if c.get('name') == part), None)
            
            if child:
                current_id = child['file_id']
                self._file_cache[current_path] = current_id
            else:
                resp = requests.post(
                    self._api_url(f"data/{current_id}/children?name={part}&type=DIR"),
                    headers=self.headers
                )
                if resp.ok:
                    new_id = resp.json().get('fileId')
                    self._file_cache[current_path] = new_id
                    current_id = new_id
                else:
                    raise IOError(f"Failed to create directory '{current_path}': {resp.text}")
        
        return current_id
    
    def _get_content(self, file_id: str, byte_range=None) -> Optional[bytes]:
        """Get file content, optionally with byte range."""
        headers = self.headers.copy()
        if byte_range:
            start, end = byte_range
            if end is not None:
                headers['Range'] = f'bytes={start}-{end-1}'
            else:
                headers['Range'] = f'bytes={start}-'
        
        resp = requests.get(
            self._api_url(f"data/{file_id}/content"), 
            headers=headers
        )
        if resp.ok:
            return resp.content
        return None
    
    def _set_content(self, path: str, content: bytes) -> None:
        """Create or update a file with the given content."""
        path = path.strip('/')
        file_id = self._resolve_path(path)
        
        if file_id:
            # Update existing file
            resp = requests.put(
                self._api_url(f"data/{file_id}/content"),
                headers={**self.headers, "Content-Type": "application/octet-stream"},
                data=content
            )
            if not resp.ok:
                raise IOError(f"Failed to update '{path}': {resp.text}")
        else:
            # Create new file
            parent_id = self._ensure_parent_dirs(path)
            filename = path.split('/')[-1]
            
            resp = requests.post(
                self._api_url(f"data/{parent_id}/children?name={filename}&type=REG"),
                headers={**self.headers, "Content-Type": "application/octet-stream"},
                data=content
            )
            if resp.ok:
                new_id = resp.json().get('fileId')
                self._file_cache[path] = new_id
            else:
                raise IOError(f"Failed to create '{path}': {resp.text}")
    
    def _delete_file(self, path: str) -> None:
        """Delete a file."""
        file_id = self._resolve_path(path)
        if not file_id:
            return  # Already doesn't exist
        
        resp = requests.delete(
            self._api_url(f"data/{file_id}"),
            headers=self.headers
        )
        if resp.ok:
            self._file_cache.pop(path, None)
        else:
            raise IOError(f"Failed to delete '{path}': {resp.text}")
    
    def _do_list_prefix(self, prefix: str) -> List[str]:
        """Recursively list all keys under a prefix."""
        prefix = prefix.strip('/')
        dir_id = self._resolve_path(prefix) if prefix else self.root_id
        if not dir_id:
            return []
        
        results = []
        children = self._list_dir(dir_id)
        for child in children:
            name = child['name']
            full_path = f"{prefix}/{name}" if prefix else name
            self._file_cache[full_path] = child['file_id']
            results.append(full_path)
            sub_children = self._list_dir(child['file_id'])
            if sub_children:
                results.extend(self._do_list_prefix(full_path))
        return results
    
    # -------------------------------------------------------------------------
    # Zarr Store Interface - Lifecycle
    # -------------------------------------------------------------------------
    
    async def _open(self) -> None:
        self._is_open = True
    
    async def _close(self) -> None:
        self._is_open = False
    
    # -------------------------------------------------------------------------
    # Zarr Store Interface - Read Operations
    # -------------------------------------------------------------------------
    
    async def get(
        self, 
        key: str, 
        prototype: BufferPrototype = default_buffer_prototype, 
        byte_range=None
    ) -> Optional[Buffer]:
        file_id = self._resolve_path(key)
        if not file_id:
            return None
        content = self._get_content(file_id, byte_range)
        if content is None:
            return None
        return prototype.buffer.from_bytes(content)
    
    async def get_partial_values(
        self, 
        prototype: BufferPrototype, 
        key_ranges: Iterable
    ) -> List[Optional[Buffer]]:
        results = []
        for key, byte_range in key_ranges:
            result = await self.get(key, prototype, byte_range)
            results.append(result)
        return results
    
    # -------------------------------------------------------------------------
    # Zarr Store Interface - Write Operations
    # -------------------------------------------------------------------------
    
    async def set(self, key: str, value: Buffer) -> None:
        if self._read_only:
            raise PermissionError("Store is read-only")
        self._set_content(key, value.to_bytes())
    
    async def set_partial_values(self, key_start_values: Iterable) -> None:
        raise NotImplementedError("Partial writes not supported")
    
    async def delete(self, key: str) -> None:
        if self._read_only:
            raise PermissionError("Store is read-only")
        self._delete_file(key)
    
    # -------------------------------------------------------------------------
    # Zarr Store Interface - Query Operations
    # -------------------------------------------------------------------------
    
    async def exists(self, key: str) -> bool:
        return self._resolve_path(key) is not None
    
    async def list(self):
        for key in self._do_list_prefix(""):
            yield key
    
    async def list_dir(self, prefix: str):
        prefix = prefix.strip('/')
        dir_id = self._resolve_path(prefix) if prefix else self.root_id
        if not dir_id:
            return
        children = self._list_dir(dir_id)
        for child in children:
            yield child['name']
    
    async def list_prefix(self, prefix: str):
        for key in self._do_list_prefix(prefix.strip('/')):
            yield key
    
    # -------------------------------------------------------------------------
    # Zarr Store Interface - Properties
    # -------------------------------------------------------------------------
    
    @property
    def supports_writes(self) -> bool:
        return not self._read_only
    
    @property  
    def supports_deletes(self) -> bool:
        return not self._read_only
    
    @property
    def supports_partial_writes(self) -> bool:
        return False
    
    @property
    def supports_listing(self) -> bool:
        return True
    
    def __eq__(self, other):
        return isinstance(other, OnedataZarrStore) and self.root_id == other.root_id
    
    def __repr__(self):
        mode = "read-only" if self._read_only else "read-write"
        return f"OnedataZarrStore(root_id={self.root_id[:30]}..., mode={mode})"


# =============================================================================
# DataHubClient - High-Level API
# =============================================================================

class DataHubClient:
    """
    High-level client for working with Zarr stores on EGI DataHub.
    
    Parameters
    ----------
    token : str, optional
        Onedata access token. If not provided, reads from DATAHUB_TOKEN 
        environment variable.
    onezone : str, optional
        Onezone hostname (default: datahub.egi.eu)
    provider : str, optional
        Default Oneprovider hostname. If not provided, will be auto-detected
        based on space.
    
    Examples
    --------
    >>> client = DataHubClient()
    >>> 
    >>> # List available spaces
    >>> spaces = client.list_spaces()
    >>> 
    >>> # Open a Zarr store
    >>> ds = client.open_zarr("Reliance/FAIR2Adapt/CS1/sample.zarr")
    >>> 
    >>> # Write a dataset
    >>> client.to_zarr(ds, "Reliance/FAIR2Adapt/CS1/output.zarr")
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        onezone: str = DEFAULT_ONEZONE,
        provider: Optional[str] = None
    ):
        self.token = token or os.environ.get("DATAHUB_TOKEN")
        if not self.token:
            raise ValueError(
                "No token provided. Set DATAHUB_TOKEN environment variable "
                "or pass token parameter."
            )
        
        self.onezone = onezone
        self.default_provider = provider
        self.headers = {"X-Auth-Token": self.token}
        
        # Cache
        self._spaces_cache: Optional[Dict[str, Dict]] = None
        self._provider_cache: Dict[str, str] = {}  # space_id -> provider
    
    def _onezone_url(self, endpoint: str) -> str:
        """Build Onezone API URL."""
        return f"https://{self.onezone}/api/v3/onezone/{endpoint}"
    
    def _oneprovider_url(self, provider: str, endpoint: str) -> str:
        """Build Oneprovider API URL."""
        return f"https://{provider}/api/v3/oneprovider/{endpoint}"
    
    # -------------------------------------------------------------------------
    # User and Space Information
    # -------------------------------------------------------------------------
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get information about the authenticated user."""
        resp = requests.get(
            self._onezone_url("user"),
            headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()
    
    def list_spaces(self, refresh: bool = False) -> Dict[str, Dict]:
        """
        List all accessible spaces.
        
        Returns
        -------
        dict
            Dictionary mapping space names to space info including:
            - space_id: The space ID
            - providers: Dict of provider IDs to storage quotas
        """
        if self._spaces_cache is not None and not refresh:
            return self._spaces_cache
        
        # Get space IDs
        resp = requests.get(
            self._onezone_url("user/effective_spaces"),
            headers=self.headers
        )
        resp.raise_for_status()
        space_ids = resp.json().get('spaces', [])
        
        # Get details for each space
        spaces = {}
        for space_id in space_ids:
            resp = requests.get(
                self._onezone_url(f"spaces/{space_id}"),
                headers=self.headers
            )
            if resp.ok:
                info = resp.json()
                name = info.get('name', space_id)
                spaces[name] = {
                    'space_id': space_id,
                    'providers': info.get('providers', {}),
                    'description': info.get('description', ''),
                }
        
        self._spaces_cache = spaces
        return spaces
    
    def get_provider_for_space(self, space_name: str) -> str:
        """
        Get the Oneprovider hostname for a space.
        
        Auto-detects the provider if not explicitly set.
        """
        if self.default_provider:
            return self.default_provider
        
        spaces = self.list_spaces()
        if space_name not in spaces:
            raise ValueError(f"Space '{space_name}' not found")
        
        space_id = spaces[space_name]['space_id']
        
        if space_id in self._provider_cache:
            return self._provider_cache[space_id]
        
        # Get first provider for this space
        providers = spaces[space_name].get('providers', {})
        if not providers:
            raise ValueError(f"No providers found for space '{space_name}'")
        
        provider_id = list(providers.keys())[0]
        
        # Get provider details
        resp = requests.get(
            self._onezone_url(f"providers/{provider_id}"),
            headers=self.headers
        )
        resp.raise_for_status()
        provider_domain = resp.json().get('domain')
        
        self._provider_cache[space_id] = provider_domain
        return provider_domain
    
    # -------------------------------------------------------------------------
    # Path Resolution
    # -------------------------------------------------------------------------
    
    def _parse_path(self, path: str) -> tuple:
        """
        Parse a path into (space_name, relative_path).
        
        Path format: "SpaceName/folder/subfolder/file.zarr"
        """
        path = path.strip('/')
        parts = path.split('/', 1)
        space_name = parts[0]
        relative_path = parts[1] if len(parts) > 1 else ""
        return space_name, relative_path
    
    def _get_space_root_id(self, space_name: str, provider: str) -> str:
        """Get the root directory ID for a space."""
        spaces = self.list_spaces()
        if space_name not in spaces:
            raise ValueError(f"Space '{space_name}' not found")
        
        space_id = spaces[space_name]['space_id']
        
        # The space root dir ID follows a specific format
        # We need to list the space to get it
        resp = requests.get(
            self._oneprovider_url(provider, f"spaces/{space_id}"),
            headers=self.headers
        )
        resp.raise_for_status()
        return resp.json().get('rootDirId') or resp.json().get('fileId')
    
    def resolve_path(self, path: str) -> tuple:
        """
        Resolve a path to (provider, file_id).
        
        Parameters
        ----------
        path : str
            Path in format "SpaceName/folder/file.zarr"
        
        Returns
        -------
        tuple
            (provider_hostname, file_id)
        """
        space_name, relative_path = self._parse_path(path)
        provider = self.get_provider_for_space(space_name)
        
        # Get space root
        root_id = self._get_space_root_id(space_name, provider)
        
        if not relative_path:
            return provider, root_id
        
        # Navigate to the target
        current_id = root_id
        for part in relative_path.split('/'):
            resp = requests.get(
                self._oneprovider_url(provider, f"data/{current_id}/children"),
                headers=self.headers
            )
            resp.raise_for_status()
            children = resp.json().get('children', [])
            child = next((c for c in children if c.get('name') == part), None)
            if not child:
                raise FileNotFoundError(f"Path not found: {path} (missing: {part})")
            current_id = child['file_id']
        
        return provider, current_id
    
    def list_directory(self, path: str) -> List[Dict[str, str]]:
        """
        List contents of a directory.
        
        Parameters
        ----------
        path : str
            Path in format "SpaceName/folder"
        
        Returns
        -------
        list
            List of dicts with 'name' and 'type' keys
        """
        provider, dir_id = self.resolve_path(path)
        
        resp = requests.get(
            self._oneprovider_url(provider, f"data/{dir_id}/children"),
            headers=self.headers
        )
        resp.raise_for_status()
        
        children = resp.json().get('children', [])
        
        # Get type info for each child
        results = []
        for child in children:
            # Get file attributes to determine type
            resp = requests.get(
                self._oneprovider_url(provider, f"data/{child['file_id']}"),
                headers=self.headers
            )
            if resp.ok:
                attrs = resp.json()
                results.append({
                    'name': child['name'],
                    'type': attrs.get('type', 'unknown'),
                    'size': attrs.get('size', 0),
                    'file_id': child['file_id']
                })
            else:
                results.append({
                    'name': child['name'],
                    'type': 'unknown',
                    'file_id': child['file_id']
                })
        
        return results
    
    # -------------------------------------------------------------------------
    # Zarr Operations
    # -------------------------------------------------------------------------
    
    def get_zarr_store(
        self, 
        path: str, 
        read_only: bool = True
    ) -> OnedataZarrStore:
        """
        Get a Zarr store for the given path.
        
        Parameters
        ----------
        path : str
            Path to Zarr store in format "SpaceName/folder/data.zarr"
        read_only : bool, optional
            If True, disable write operations (default: True)
        
        Returns
        -------
        OnedataZarrStore
            A Zarr v3 compatible store
        """
        provider, file_id = self.resolve_path(path)
        return OnedataZarrStore(
            root_file_id=file_id,
            token=self.token,
            provider=provider,
            read_only=read_only
        )
    
    def open_zarr(self, path: str, **kwargs):
        """
        Open a Zarr store as an xarray Dataset.
        
        Parameters
        ----------
        path : str
            Path to Zarr store in format "SpaceName/folder/data.zarr"
        **kwargs
            Additional arguments passed to xr.open_zarr()
        
        Returns
        -------
        xarray.Dataset
            The opened dataset
        """
        if not HAS_XARRAY:
            raise ImportError("xarray is required for open_zarr()")
        
        store = self.get_zarr_store(path, read_only=True)
        
        # Set defaults for Zarr v3
        kwargs.setdefault('consolidated', False)
        kwargs.setdefault('zarr_format', 3)
        
        return xr.open_zarr(store, **kwargs)
    
    def create_zarr_store(self, path: str) -> OnedataZarrStore:
        """
        Create a new Zarr store directory and return a writable store.
        
        Parameters
        ----------
        path : str
            Path for new Zarr store in format "SpaceName/folder/data.zarr"
        
        Returns
        -------
        OnedataZarrStore
            A writable Zarr store
        """
        space_name, relative_path = self._parse_path(path)
        provider = self.get_provider_for_space(space_name)
        
        # Navigate to parent and create the zarr directory
        parent_path = '/'.join(relative_path.split('/')[:-1])
        zarr_name = relative_path.split('/')[-1]
        
        if parent_path:
            full_parent = f"{space_name}/{parent_path}"
            _, parent_id = self.resolve_path(full_parent)
        else:
            parent_id = self._get_space_root_id(space_name, provider)
        
        # Create the zarr directory
        resp = requests.post(
            self._oneprovider_url(provider, f"data/{parent_id}/children?name={zarr_name}&type=DIR"),
            headers=self.headers
        )
        
        if resp.status_code == 409:
            # Already exists, get its ID
            resp = requests.get(
                self._oneprovider_url(provider, f"data/{parent_id}/children"),
                headers=self.headers
            )
            resp.raise_for_status()
            children = resp.json().get('children', [])
            existing = next((c for c in children if c['name'] == zarr_name), None)
            if existing:
                zarr_id = existing['file_id']
            else:
                raise IOError(f"Failed to find or create {path}")
        elif resp.ok:
            zarr_id = resp.json().get('fileId')
        else:
            raise IOError(f"Failed to create {path}: {resp.text}")
        
        return OnedataZarrStore(
            root_file_id=zarr_id,
            token=self.token,
            provider=provider,
            read_only=False
        )
    
    def to_zarr(self, dataset, path: str, **kwargs):
        """
        Write an xarray Dataset to a Zarr store on DataHub.
        
        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset to write
        path : str
            Path for Zarr store in format "SpaceName/folder/data.zarr"
        **kwargs
            Additional arguments passed to ds.to_zarr()
        
        Examples
        --------
        >>> client.to_zarr(ds, "Reliance/FAIR2Adapt/CS1/output.zarr")
        """
        if not HAS_XARRAY:
            raise ImportError("xarray is required for to_zarr()")
        
        store = self.create_zarr_store(path)
        
        # Set defaults for Zarr v3
        kwargs.setdefault('zarr_format', 3)
        kwargs.setdefault('consolidated', False)
        kwargs.setdefault('mode', 'w')
        
        dataset.to_zarr(store, **kwargs)
        print(f"✅ Written to: {path}")
    
    def delete_zarr(self, path: str) -> None:
        """
        Delete a Zarr store.
        
        Parameters
        ----------
        path : str
            Path to Zarr store to delete
        """
        provider, file_id = self.resolve_path(path)
        
        resp = requests.delete(
            self._oneprovider_url(provider, f"data/{file_id}"),
            headers=self.headers
        )
        
        if not resp.ok:
            raise IOError(f"Failed to delete {path}: {resp.text}")
        
        print(f"✅ Deleted: {path}")


# =============================================================================
# Convenience Functions
# =============================================================================

def open_zarr(path: str, token: Optional[str] = None, **kwargs):
    """
    Open a Zarr store from EGI DataHub.
    
    Parameters
    ----------
    path : str
        Path in format "SpaceName/folder/data.zarr"
    token : str, optional
        Onedata token (uses DATAHUB_TOKEN env var if not provided)
    **kwargs
        Additional arguments passed to xr.open_zarr()
    
    Returns
    -------
    xarray.Dataset
    
    Examples
    --------
    >>> ds = open_zarr("Reliance/FAIR2Adapt/CS1/sample.zarr")
    """
    client = DataHubClient(token=token)
    return client.open_zarr(path, **kwargs)


def to_zarr(dataset, path: str, token: Optional[str] = None, **kwargs):
    """
    Write an xarray Dataset to EGI DataHub.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to write
    path : str
        Path in format "SpaceName/folder/data.zarr"
    token : str, optional
        Onedata token (uses DATAHUB_TOKEN env var if not provided)
    **kwargs
        Additional arguments passed to ds.to_zarr()
    
    Examples
    --------
    >>> to_zarr(ds, "Reliance/FAIR2Adapt/CS1/output.zarr")
    """
    client = DataHubClient(token=token)
    client.to_zarr(dataset, path, **kwargs)


# =============================================================================
# CLI Support
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EGI DataHub Zarr Toolkit")
    parser.add_argument("command", choices=["list", "info", "spaces"],
                       help="Command to run")
    parser.add_argument("path", nargs="?", default="",
                       help="Path (for list command)")
    parser.add_argument("--token", "-t", help="Onedata access token")
    
    args = parser.parse_args()
    
    client = DataHubClient(token=args.token)
    
    if args.command == "spaces":
        spaces = client.list_spaces()
        print("Available spaces:")
        for name, info in spaces.items():
            print(f"  - {name}")
    
    elif args.command == "list":
        if not args.path:
            # List spaces
            spaces = client.list_spaces()
            for name in spaces:
                print(f"📁 {name}/")
        else:
            # List directory
            items = client.list_directory(args.path)
            for item in items:
                icon = "📁" if item['type'] == 'DIR' else "📄"
                print(f"{icon} {item['name']}")
    
    elif args.command == "info":
        user = client.get_user_info()
        print(f"User: {user.get('name', 'Unknown')}")
        print(f"User ID: {user.get('userId', 'Unknown')}")
