# spaTScope

## Overview
spaTScope is a web application for visual exploration of geolocated time series. It allows users to visually explore large collections of geolocated time series and obtain insights about trends and patterns in their area of interest. spaTScope leverages a hybrid index that allows users to navigate and group the available time series based not only on their similarity but also on spatial proximity. It was introduced in the following paper:

G. Chatzigeorgakidis, K. Patroumpas, D. Skoutas and S. Athanasiou, "A Visual Explorer for Geolocated Time Series (Demo Paper)", In Proceedings of the 28th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, 2020.

## Installation and Usage
The application can be executed by first initiating the the back-end via the following command:
```
java -jar btsrindex-0.0.1-SNAPSHOT.jar
```

Then, on a terminal located at the parent directory of the main.py file, issue the following command:
```
bokeh serve --show spaTScope
```

Be sure to have all necessary dependencies found in main.py installed.

The input dataset file should be a CSV file containing geolocated time series of equal length in a per line fashion, as follows:
```
geoTS_id1, longitude1, latitude1, ts1_value1, ts1_value2, ..., ts1_valueN
geoTS_id2, longitude2, latitude2, ts2_value1, ts2_value2, ..., ts2_valueN
.
.
.
geoTS_idM, longitudeM, latitudeM, tsM_value1, tsM_value2, ..., tM_valueN
```

Specifically:\
Column 1: The id/name of each geolocated time series\
Columns 2 and 3: The coordinates (longitude and latitude, in EPSG:3627 format)\
Subsequent columns: The time series values (the time series should be of equal length and time-aligned)

## License
The contents of this project are licensed under the [Apache License 2.0](https://github.com/smartdatalake/simsearch/blob/master/LICENSE).

## Acknowledgement
This software is being developed in the context of the [SmartDataLake](https://smartdatalake.eu/) project. This project has received funding from the European Union’s [Horizon 2020 research and innovation programme](https://ec.europa.eu/programmes/horizon2020/en) under grant agreement No 825041.
