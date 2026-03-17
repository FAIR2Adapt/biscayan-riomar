# biscayan-riomar

[![RoHub Executable Research Object](https://img.shields.io/badge/RoHub-FAIR_Executable_Research_Object-2ea44f?logo=Open+Access&logoColor=blue)](https://w3id.org/ro-id/ea3ac578-ab0a-47c8-a10a-9d06678a9450)


RiOMar Project – Coastal Water Quality Anticipation to manage coastal zone ecosystem responses for biodiversity conservation


To copy in CS2 on EGI datahub:

```
export CS2_ID="000000000052F80F67756964236165626335306634396136633664373236373264393765396232346130653162636862343531233732356634616233366362323664306662666330633132346337373565666565636865653439"

curl -X POST "https://cesnet-oneprovider-01.datahub.egi.eu/api/v3/oneprovider/data/${CS2_ID}/children?name=weather.nc&type=REG" -H "X-Auth-Token:${DATAHUB_TOKEN}" -H "Content-Type:application/octet-stream" --data-binary @weather.nc

```


where DATAHUB_TOKEN is a REST and CDMI token. 
