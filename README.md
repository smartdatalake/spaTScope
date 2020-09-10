# spaTScope
This an early preview version of the interactive visual explorer for geolocated time series introduced in the following paper (to be published):

G. Chatzigeorgakidis, K. Patroumpas, D. Skoutas and S. Athanasiou, "A Visual Explorer for Geolocated Time Series (Demo Paper)", In Proceedings of the 28th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, 2020.

The application can be executed by first initiating the the back-end via the following command:
```
java -jar btsrindex-0.0.1-SNAPSHOT.jar "path_to_input_dataset/input_dataset.csv"
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
Columns 2 and 3: The coordinates (longitude and latitude, in WGS84 format)\
Subsequent columns: The time series values (the time series should be of equal length and time-aligned)
