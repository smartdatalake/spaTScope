from __future__ import print_function

from bokeh.plotting import figure, curdoc
from bokeh.tile_providers import STAMEN_TERRAIN_RETINA, get_provider
from bokeh.models import ColumnDataSource, WheelZoomTool, Band, Button, LabelSet, Dropdown, TapTool
from bokeh.layouts import column, row
from bokeh.events import ButtonClick, Tap

import pandas as pd
from io import StringIO
import numpy as np
import json
import requests
from shapely import wkt
import geopandas
from colour import Color
from pyproj import Transformer
from types import FunctionType
from copy import copy

####################################################################################
################################# GLOBAL VARIABLES #################################
####################################################################################
service_url = 'http://localhost:8080/'
catalog_url = service_url + 'runQuery'
default_selected_bundle = 0
max_k1 = 20 + 1
default_k1 = 5
default_k2 = 5
default_query_x1 = -8240005.4443985885
default_query_x2 = -8228804.847202021
default_query_y1 = 4973643.0294881
default_query_y2 = 4981584.8322417
default_scale = 15000
query_x1 = 0
query_x2 = 0
query_y1 = 0
query_y2 = 0
geoms = 0
bundles = 0
####################################################################################
####################################################################################

####################################################################################
################################# SOME FUNCTIONS ###################################
####################################################################################
def get_scale():
	global query_x1, query_x2
	x_range = default_query_x2-default_query_x1
	new_x_range = query_x2-query_x1
	new_scale = (new_x_range * default_scale) / x_range
	return new_scale

def get_query_from_scale(new_scale):
	global default_query_x1, default_query_x2, default_query_y1, default_query_y2
	x_range = abs(default_query_x2-default_query_x1)
	new_x_range = (x_range * new_scale) / default_scale
	diff_x = abs(x_range - new_x_range)
	if new_x_range > x_range:
		new_query_x1 = default_query_x1 - (diff_x/2)
		new_query_x2 = default_query_x2 + (diff_x/2)
	else:
		new_query_x1 = default_query_x1 + (diff_x/2)
		new_query_x2 = default_query_x2 - (diff_x/2)

	y_range = abs(default_query_y2-default_query_y1)
	new_y_range = (y_range * new_scale) / default_scale
	diff_y = abs(y_range - new_y_range)
	if new_y_range > y_range:
		new_query_y1 = default_query_y1 - (diff_y/2)
		new_query_y2 = default_query_y2 + (diff_y/2)
	else:
		new_query_y1 = default_query_y1 + (diff_y/2)
		new_query_y2 = default_query_y2 - (diff_y/2)

	return new_query_x1, new_query_x2, new_query_y1, new_query_y2

def lon_to_web_mercator(lon):
    k = 6378137
    return lon * (k * np.pi / 180.0)

def lat_to_web_mercator(lat):
    k = 6378137
    return np.log(np.tan((90 + lat) * np.pi / 360.0)) * k

def wgs84_to_web_mercator(df, lon="lon", lat="lat"):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi / 180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi / 360.0)) * k
    return df

def epsg3627_to_web_mercrator(coords):
    transformer = Transformer.from_crs("epsg:3627", "epsg:4326")
    x1, y1 = coords[0], coords[1]
    x2, y2, = transformer.transform(x1, y1)
    x2 = lat_to_web_mercator(x2)
    y2 = lon_to_web_mercator(y2)
    return (y2, x2)

def get_spatial_data(selected_bundle):
    global geoms, bundles, default_k1
    centroidsX = []
    centroidsY = []
    areas = []
    cols = []
    counts = []
    geometries = []
    top = []
    bottom = []
    left = []
    right = []
    selected_geoms = geoms.loc[geoms['bundle'] == selected_bundle].reset_index(drop=True)
    for i in list(range(0, len(selected_geoms['geometries']))):
        X = lon_to_web_mercator(selected_geoms['geometries'][i].centroid.coords[0][0])
        Y = lat_to_web_mercator(selected_geoms['geometries'][i].centroid.coords[0][1])
        if (X not in centroidsX) and (Y not in centroidsY):
            centroidsX.append(X)
            centroidsY.append(Y)
            geometries.append(selected_geoms['geometries'][i].bounds)
            
            top_coord = lat_to_web_mercator(selected_geoms['geometries'][i].bounds[3])
            bottom_coord = lat_to_web_mercator(selected_geoms['geometries'][i].bounds[1])
            left_coord = lon_to_web_mercator(selected_geoms['geometries'][i].bounds[0])
            right_coord = lon_to_web_mercator(selected_geoms['geometries'][i].bounds[2])
          
            if (top_coord == bottom_coord and left_coord == right_coord):
                top_coord += 10
                bottom_coord -= 10
                left_coord -= 10
                right_coord += 10
                
            elif (top_coord - bottom_coord < 5):
                top_coord += 50
                bottom_coord -= 50

            elif (left_coord - right_coord < 5):
                left_coord -= 50
                right_coord += 50

            top.append(top_coord)
            bottom.append(bottom_coord)
            left.append(left_coord)
            right.append(right_coord)
        else:
            continue
        counts.append(selected_geoms['count'][i])
        areas.append(selected_geoms['geometries'][i].area)

    multiplier = 200
    try:
    	areas = [multiplier*(x / max(areas)) for x in areas]
    except:
    	areas = [100]
    	default_k1 = len(geoms)
    	noOfTSsummaries.label = str(default_k1) + " bundles"
    #cols = [str(colors[int(x*(49/2000)*10)]) for x in areas]
    cols = []
    for j in range(len(areas)):
    	col = (counts[j]/areas[j])*(49/500)*100
    	if col > 49:
    		col = 49
    	cols.append(str(colors[int(col)]))
    return counts, centroidsY, centroidsX, areas, cols, geometries, top, bottom, left, right

def createFunc(i):
    def callback(but):
	    global default_selected_bundle
	    for b in buttons:
	    	b.button_type = "success"
	    but.button_type = "primary"
	    bundle = int(but.label.split(" ")[2])-1
	    prev = p.select_one({'name' : str(default_selected_bundle)})
	    prev_labels = p.select_one({'name' : "label " + str(default_selected_bundle)})
	    prev.visible = False
	    prev_labels.visible = False
	    curr = p.select_one({'name' : str(bundle)})
	    curr_labels = p.select_one({'name' : "label " + str(bundle)})
	    curr.visible = True
	    curr_labels.visible = True
	    default_selected_bundle = bundle
    return callback

############################
#### CALLBACK FUNCTIONS ####
############################
def update_plot():
	# Create the query
	global query_x1, query_x2, query_y1, query_y2, service_url, catalog_url, default_k1
	transformer = Transformer.from_crs("epsg:3857", "epsg:3627")
	qx1, qy1 = transformer.transform(query_x1, query_y1)
	qx2, qy2 = transformer.transform(query_x2, query_y2)

	# Get the data from the server via REST API
	global geoms, bundles
	config = dict()
	config['x1'] = str(qx1)
	config['x2'] = str(qx2)
	config['y1'] = str(qy1)
	config['y2'] = str(qy2)
	config['k1'] = str(default_k1)
	config['k2'] = str(default_k2)
	config = json.dumps(config)
	config = json.loads(config)
	r = requests.post(catalog_url, json=config)
	response = r.json()
	geoms = pd.read_csv(StringIO(response["content"][0]), header=0, delimiter='|')
	bundles = pd.read_csv(StringIO(response["content"][1]), header=0, delimiter=';')

	# Update map summaries
	my_df = []
	for i in list(range(0, len(geoms['wkt']))):
	    d = {
	        'bundle' : geoms['bundle'][i],
	        'count' : geoms['count'][i],
	        'geometries' : geoms['wkt'][i]
	    }
	    my_df.append(d)
	geoms = pd.DataFrame(my_df)
	if len(geoms) > 0:
		geoms['geometries'] = geoms['geometries'].apply(wkt.loads)
		geoms = geopandas.GeoDataFrame(geoms, geometry='geometries')
		geoms.crs = "EPSG:3627"
		geoms = geoms.to_crs(crs="EPSG:4326")

	# Update time series summaries
	get_spatial_data(i)
	if len(geoms) > 0: # if we have obtained some results, continue normally
		for i in list(range(0, default_k1)):
		    counts, centroidsY, centroidsX, areas, cols, geometries, top, bottom, left, right = get_spatial_data(i)
		    new_data=dict(counts=counts,
		                  lat=centroidsY,
		                  lon=centroidsX,
		                  size=areas,
		                  colors=cols,
		                  geometries=geometries,
		                  top=top,
		                  bottom=bottom,
		                  left=left,
		                  right=right)
		    sourcesMap[i].data = new_data
		    counts = new_data['counts']

		    upper = [float(i) for i in bundles['UPPER_BOUND'][i].split(",")]
		    lower = [float(i) for i in bundles['LOW_BOUND'][i].split(",")]
		    average = [(g + h) / 2 for g, h in zip(upper, lower)]
		    new_data=dict(upper=upper,
		                  lower=lower,
		                  average=average,
		                  timestamps=list(range(0, len(bundles['UPPER_BOUND'][0].split(",")))))
		    sourcesBundles[i].data = new_data
		    if (upper[0] < float(-1000000000) or lower[0] > float(1000000000)):
		    	ts_plot = ts_plots.select_one({'name' : "ts_plot " + str(i)})
		    	ts_plot.visible = False
		    elif len(counts) == 0:
		    	ts_plot = ts_plots.select_one({'name' : "ts_plot " + str(i)})
		    	ts_plot.visible = False
		    else:
		    	ts_plot = ts_plots.select_one({'name' : "ts_plot " + str(i)})
		    	ts_plot.visible = True
	else: # if results are empty, just plot nothing
		for i in list(range(0, default_k1)):
			new_data=dict(counts=[],
			              lat=[],
			              lon=[],
			              size=[],
			              colors=[],
		                  geometries=[])
			sourcesMap[i].data = new_data
			ts_plot = ts_plots.select_one({'name' : "ts_plot " + str(i)})
			ts_plot.visible = False

	ts_plot = ts_plots.select_one({'name' : "ts_plot " + str(20)})
	ts_plot.visible = True
	for i in list(range(default_k1, max_k1)):
		ts_plot = ts_plots.select_one({'name' : "ts_plot " + str(i)})
		ts_plot.visible = False

	# Update the scale button
	new_scale = get_scale()
	scaleSelect.label = "1:" + str(int(new_scale))

# Other callbacks   
def selected_k1(event):
	global default_k1
	default_k1 = int(event.item)
	noOfTSsummaries.label = str(default_k1) + " bundles"
	update_plot()

def selected_scale(event):
	new_scale = int(event.item.split(":")[1])
	new_query_x1, new_query_x2, new_query_y1, new_query_y2 = get_query_from_scale(new_scale)
	map = curdoc().select_one({'name' : "map"})
	map.x_range.start = new_query_x1
	map.x_range.end = new_query_x2
	map.y_range.start = new_query_y1
	map.y_range.end = new_query_y2
	global query_x1, query_x2, query_y1, query_y2
	query_x1 = new_query_x1
	query_x2 = new_query_x2
	query_y1 = new_query_y1
	query_y2 = new_query_y2
	update_plot()

def reset_plot(event):
	global query_x1, query_x2, query_y1, query_y2, default_k1
	map = curdoc().select_one({'name' : "map"})
	map.x_range.start = default_query_x1
	map.x_range.end = default_query_x2
	map.y_range.start = default_query_y1
	map.y_range.end = default_query_y2
	default_k1 = 5
	noOfTSsummaries.label = str(default_k1) + " bundles"
	query_x1 = default_query_x1
	query_x2 = default_query_x2
	query_y1 = default_query_y1
	query_y2 = default_query_y2
	for b in buttons:
	    b.button_type = "success"
	buttons[0].button_type = "primary"
	update_plot()

def getCallback(calls, i):
	return lambda: calls[i](buttons[i])

def update1(attr,new,old):
    global query_x1
    query_x1 = new

def update2(attr,new,old):
    global query_x2
    query_x2 = new

def update3(attr,new,old):
    global query_y1
    query_y1 = new

def update4(attr,new,old):
    global query_y2
    query_y2 = new

def selected_circle(event):
	global query_x1, query_x2, query_y1, query_y2
	coords = sourcesMap[default_selected_bundle].data['geometries'][sourcesMap[default_selected_bundle].selected.indices[0]]
	c_query_x1 = lon_to_web_mercator(coords[0])
	c_query_x2 = lon_to_web_mercator(coords[2])
	c_query_y1 = lat_to_web_mercator(coords[1])
	c_query_y2 = lat_to_web_mercator(coords[3])
	map = curdoc().select_one({'name' : "map"})

	c_x_diff = c_query_x2 - c_query_x1
	c_y_diff = c_query_y2 - c_query_y1
	x_diff = query_x2 - query_x1
	y_diff = query_y2 - query_y1

	if c_x_diff > c_y_diff:
		scale_f = c_x_diff/x_diff
		c_y_diff = y_diff*scale_f
		map.x_range.start = c_query_x1
		map.x_range.end = c_query_x2
		map.y_range.start = query_y1 + ((y_diff-c_y_diff)/2)
		map.y_range.end = query_y2 - ((y_diff-c_y_diff)/2)
		query_x1 = c_query_x1
		query_x2 = c_query_x2
		query_y1 = query_y1 + ((y_diff-c_y_diff)/2)
		query_y2 = query_y2 - ((y_diff-c_y_diff)/2)
		update_plot()
	else:
		scale_f = c_y_diff/y_diff
		c_x_diff = x_diff*scale_f
		map.x_range.start = query_x1 + ((x_diff-c_x_diff)/2)
		map.x_range.end = query_x2 - ((x_diff-c_x_diff)/2)
		map.y_range.start = c_query_y1
		map.y_range.end = c_query_y2
		query_x1 = query_x1 + ((x_diff-c_x_diff)/2)
		query_x2 = query_x2 - ((x_diff-c_x_diff)/2)
		query_y1 = c_query_y1
		query_y2 = c_query_y2
		update_plot()
################################################################
####################################################################################


####################################################################################
########################## VISUALIZATION INITIALIZATION ############################
####################################################################################

# Generate the color list
red = Color("green")
colors = list(red.range_to(Color("red"), 50))

# Create the map plot for the default scale
tile_provider = get_provider(STAMEN_TERRAIN_RETINA)
Manhattan = x_range, y_range = ((default_query_x1, default_query_x2), (default_query_y1, default_query_y2))
p = figure(x_range=x_range, y_range=y_range, x_axis_type="mercator", y_axis_type="mercator", name="map", tools="tap, pan, wheel_zoom")
p.on_event(Tap, selected_circle)

p.x_range.on_change('start', update1)
p.x_range.on_change('end', update2)
p.y_range.on_change('start', update3)
p.y_range.on_change('end', update4)

p.sizing_mode = 'stretch_both'
p.add_tile(tile_provider)
p.xaxis.visible = False
p.yaxis.visible = False
p.toolbar.active_scroll = p.select_one(WheelZoomTool) 
p.toolbar_location = None

# Create the time series plots for the default map scale
ts_plots = []
buttons = []
sourcesMap = []
sourcesBundles = []
ts_panel_width = 500
ts_plot_height = 200

for i in list(range(0, max_k1)):
    source = ColumnDataSource(
        data=dict(counts=[],
                  lat=[],
                  lon=[],
                  size=[],
                  colors=[],
		          geometries=[],
		          top=[],
		          bottom=[],
		          left=[],
		          right=[])
    )
    sourcesMap.append(source)
    #c = p.circle(x="lon", y="lat", size="size", fill_color="colors", fill_alpha=0.6, line_color="colors", line_width=4, source=source, name=str(i))
    c = p.quad(top="top", bottom="bottom", left="left", right="right", fill_color="colors", fill_alpha=0.6, line_color="colors", line_width=4, source=source, name=str(i))

    labels = LabelSet(x='lon', y='lat', text='counts', text_font_size='16px', text_font_style='bold', level='overlay', x_offset=-7, y_offset=-7, source=source, render_mode='canvas', name="label " + str(i))
    p.add_layout(labels)

    if i==0:
        c.visible = True
        labels.visible = True
    else:
        c.visible = False
        labels.visible = False

    source = ColumnDataSource(
        data=dict(upper=[],
                  lower=[],
                  average=[],
                  timestamps=[])
    )
    sourcesBundles.append(source)
    s = figure(background_fill_color="#fafafa", name="fig " + str(i))
    s.toolbar_location = None
    s.line(x="timestamps", y="upper", color="#53777a", line_width=3, source=source, alpha=0.3)
    s.line(x="timestamps", y="lower", color="#53777a", line_width=3, source=source, alpha=0.3)
    s.line(x="timestamps", y="average", color="#53777a", line_width=2, source=source)
    band = Band(base="timestamps", upper='upper', lower='lower', source=source, level='underlay', fill_alpha=0.2, fill_color='#53777a')
    s.add_layout(band)
    buttons.append(Button(label="Show Bundle " + str(i+1), button_type="success", name="button " + str(i)))
    s.height = ts_plot_height
    s.width = ts_panel_width
    layout = column(buttons[i], s, name="ts_plot " + str(i))
    ts_plots.append(layout)
##################################################################
##################################################################

# Add all extra cuntionality buttons and define their callbacks
fetchButton = Button(label="Fetch", button_type="primary")
fetchButton.on_event(ButtonClick, update_plot)

resetButton = Button(label="Reset", button_type="danger")
resetButton.on_event(ButtonClick, reset_plot)

menuTSsummaries = [("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"), ("6", "6"), ("7", "7"), ("8", "8"), ("9", "9"), ("10", "10")]
noOfTSsummaries = Dropdown(button_type="warning", menu=menuTSsummaries)
noOfTSsummaries.label = str(default_k1) + " bundles"
noOfTSsummaries.on_click(selected_k1)

menuDatasets = [("New York City", "New York City"), ("Alicante", "Alicante"), ("Open File...", "Open File...")]
datasetSelect = Dropdown(label="NYC", button_type="warning", menu=menuDatasets)

menuScale = [("1:500", "1:500"), ("1:5000", "1:5000"), ("1:10000", "1:10000"), ("1:15000", "1:15000"), ("1:20000", "1:20000"), ("1:25000", "1:25000")]
scaleSelect = Dropdown(button_type="warning", menu=menuScale)
scaleSelect.label = "1:" + str(default_scale)
scaleSelect.on_click(selected_scale)

callbacks = [createFunc(i) for i in range(max_k1)]
for i in range(max_k1):
	buttons[i].on_click(getCallback(callbacks, i))
buttons[0].button_type = "primary"

# Add everything to layouts and to the final application
ts_plots = column(ts_plots)
extra_func_buttons1 = row(fetchButton, resetButton)
extra_func_buttons2 = row(datasetSelect, noOfTSsummaries, scaleSelect)
extra_func_buttons1.width = ts_panel_width
extra_func_buttons2.width = ts_panel_width
lay = column(extra_func_buttons1, extra_func_buttons2, ts_plots, name="ts_plots")
curdoc().add_root(lay)
curdoc().add_root(p)
curdoc().title = "spaTScope"

# Render the visualization
query_x1 = default_query_x1
query_x2 = default_query_x2
query_y1 = default_query_y1
query_y2 = default_query_y2
update_plot()