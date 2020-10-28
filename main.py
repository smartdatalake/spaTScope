from __future__ import print_function

from bokeh.plotting import figure, curdoc
from bokeh.tile_providers import STAMEN_TERRAIN_RETINA, get_provider
from bokeh.models import ColumnDataSource, WheelZoomTool, Band, Button, LabelSet, Dropdown, TapTool, Quad
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

import tkinter as tk
from tkinter import ttk, StringVar
from tkinter.filedialog import askopenfilename
import tkinter.simpledialog
from tkinter import Toplevel, Label
import threading
import time

####################################################################################
################################# GLOBAL VARIABLES #################################
####################################################################################
service_url = 'http://localhost:8080/'
query_url = service_url + 'runQuery'
build_index_url = service_url + 'buildIndex'
get_rootMBR_url = service_url + 'getRootMBR'
default_selected_bundle = 0
max_k1 = 26
max_bundles = 10
default_k1 = 5
default_k2 = 5
initial_query_x1 = -18823760.30596319
initial_query_x2 = 21410224.676255226
initial_query_y1 = -8443745.818059208
initial_query_y2 = 19211608.358907666
default_query_x1 = -18823760.30596319
default_query_x2 = 21410224.676255226
default_query_y1 = -8443745.818059208
default_query_y2 = 19211608.358907666
default_scale = 15000
query_x1 = 0
query_x2 = 0
query_y1 = 0
query_y2 = 0
geoms = 0
bundles = 0
window = None
message = None
response = None
filename = None
cancelled = True
local_entry = None
remote_entry = None
####################################################################################
####################################################################################

####################################################################################
##################################### CLASSES ######################################
####################################################################################
class show_message(Toplevel):
	def __init__(self, master, positionRight, positionDown):
		Toplevel.__init__(self, master)
		master.withdraw()
		self.title("Loading...")
		label = tk.Label(self, text="Please Wait...")
		label.pack(side="top", fill="both", expand=True, padx=30, pady=20)
		self.geometry("+{}+{}".format(positionRight+500, positionDown))
		self.lift()
		self.attributes('-topmost',True)
		self.after_idle(self.attributes, '-topmost',False)

class GUI_local:
	def __init__(self, window): 
		global input_text1, input_text2, filename, cancelled, local_entry, remote_entry
		cancelled = True
		self.path = ''

		window.title("Open Dataset...")
		window.resizable(0, 0) # this prevents from resizing the window
		window.geometry("763x35")
		
		local_entry = ttk.Entry(window, width = 70)
		local_entry.grid(row = 0, column = 0, ipadx=5, ipady=4)
		ttk.Button(window, text = "Browse", command = lambda: self.set_path_local_field()).grid(row = 0, column=1, ipadx=5, ipady=5)
		local = ttk.Button(window, text = "OK", command = lambda: self.get_filepath()).grid(row = 0, column=2, ipadx=5, ipady=5)

	def set_path_local_field(self):
		global local_entry, window, input_text1
		self.path = askopenfilename(filetypes=[("CSV files", ".csv")], parent=window)
		local_entry.delete(0, "end")
		local_entry.insert(0, self.path)

	def get_filepath(self):
		global window, filename, cancelled, local_entry
		cancelled = False
		filename = local_entry.get()
		window.destroy()

class GUI_remote:
	def __init__(self, window): 
		global input_text1, input_text2, filename, cancelled, local_entry, remote_entry
		cancelled = True
		self.path = ''

		window.title("Open Dataset...")
		window.resizable(0, 0) # this prevents from resizing the window
		window.geometry("673x35")

		remote_entry = ttk.Entry(window, width = 70)
		remote_entry.grid(row = 0, column = 0, ipadx=5, ipady=4)
		remote = ttk.Button(window, text = "OK", command = lambda: self.get_filepath()).grid(row = 0, column=1, ipadx=5, ipady=5)

	def get_filepath(self):
		global window, filename, cancelled, remote_entry
		cancelled = False
		filename = remote_entry.get()
		window.destroy()

####################################################################################
#################################### FUNCTIONS #####################################
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
        counts.append(" " + str(selected_geoms['count'][i]) + " ")
        areas.append(selected_geoms['geometries'][i].area)

    multiplier = 200
    try:
    	areas = [multiplier*(x / max(areas)) for x in areas]
    except:
    	areas = [100]
    	default_k1 = len(geoms)
    	noOfTSsummaries.label = str(default_k1) + " bundles"
    
	# Generate the colors
    red = Color("green")
    colors = list(red.range_to(Color("red"), len(areas)+1))
    cols = []
    for j in range(len(areas)):
    	col = (float(counts[j])/areas[j])*((len(areas))/500)*100
    	if col > len(areas):
    		col = len(areas)
    	cols.append(str(colors[int(col)]))
    return counts, centroidsY, centroidsX, areas, cols, geometries, top, bottom, left, right

def createFunc(i):
    def callback(but):
	    global default_selected_bundle
	    for b in buttons:
	    	b.css_classes =['custom_button']
	    but.css_classes = ['custom_button_selected']
	    bundle = int(but.label)-1
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
def fecth_data():
	global default_k1, default_selected_bundle
	default_k1 = 5
	noOfTSsummaries.label = str(default_k1) + " bundles"
	for i in list(range(0, max_k1)):
		mbrs = curdoc().select_one({'name' : str(i)})
		labels = curdoc().select_one({'name' : "label " + str(i)})
		if i==0:
			mbrs.visible = True
			labels.visible = True
		else:
			mbrs.visible = False
			labels.visible = False
	for b in buttons:
	    b.css_classes=['custom_button']
	buttons[0].css_classes=['custom_button_selected']
	bundle = int(buttons[0].label)-1
	prev = p.select_one({'name' : str(default_selected_bundle)})
	prev_labels = p.select_one({'name' : "label " + str(default_selected_bundle)})
	prev.visible = False
	prev_labels.visible = False
	curr = p.select_one({'name' : str(bundle)})
	curr_labels = p.select_one({'name' : "label " + str(bundle)})
	curr.visible = True
	curr_labels.visible = True
	default_selected_bundle = bundle
	update_plot()

def update_plot():
	# Create the query
	global query_x1, query_x2, query_y1, query_y2, service_url, query_url, default_k1, max_bundles
	transformer = Transformer.from_crs("epsg:3857", "epsg:3627")
	qx1, qy1 = transformer.transform(query_x1+150, query_y1+150)
	qx2, qy2 = transformer.transform(query_x2-150, query_y2-150)

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
	r = requests.post(query_url, json=config)
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
	if len(geoms) > 0: # if we have obtained some results, continue normally
		for i in list(range(0, len(bundles))):
		    counts, centroidsY, centroidsX, areas, cols, geometries, top, bottom, left, right = get_spatial_data(i)
		    fill_alpha = []
		    line_alpha = []
		    if len(geoms) == len(bundles) and len(bundles) > max_bundles:
		    	counts = []
		    	cols = []
		    	for j in range(len(areas)):
		    		fill_alpha.append(1)
		    		line_alpha.append(1)
		    		cols.append("blue")
		    		counts.append("")
		    else:
		    	for j in range(len(counts)):
		    		fill_alpha.append(0.25)
		    		line_alpha.append(0.75)

		    new_data=dict(counts=counts,
		                  lat=centroidsY,
		                  lon=centroidsX,
		                  size=areas,
		                  fill_alpha=fill_alpha,
		                  line_alpha=line_alpha,
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
	scaleSelect.label = "Scale 1:" + str(int(new_scale))

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
	global default_k1, default_selected_bundle
	default_k1 = 5
	noOfTSsummaries.label = str(default_k1) + " bundles"
	for i in list(range(0, max_k1)):
		mbrs = curdoc().select_one({'name' : str(i)})
		labels = curdoc().select_one({'name' : "label " + str(i)})
		if i==0:
			mbrs.visible = True
			labels.visible = True
		else:
			mbrs.visible = False
			labels.visible = False
	for b in buttons:
	    b.css_classes=['custom_button']
	buttons[0].css_classes=['custom_button_selected']
	bundle = int(buttons[0].label)-1
	prev = p.select_one({'name' : str(default_selected_bundle)})
	prev_labels = p.select_one({'name' : "label " + str(default_selected_bundle)})
	prev.visible = False
	prev_labels.visible = False
	curr = p.select_one({'name' : str(bundle)})
	curr_labels = p.select_one({'name' : "label " + str(bundle)})
	curr.visible = True
	curr_labels.visible = True
	default_selected_bundle = bundle
	update_plot()

def threaded_function():
	global message, window, response
	config = dict()
	config['filename'] = filename
	config = json.dumps(config)
	config = json.loads(config)
	r = requests.post(build_index_url, json=config)
	response = r.json()
	message.destroy()
	window.destroy()

def selected_dataset(event):
	global query_x1, query_x2, query_y1, query_y2, default_query_x1, default_query_x2, default_query_y1, default_query_y2, window, message, filename, response, build_index_url, cancelled

	window = tkinter.Tk()
	if event.item == "Open Local":
		gui = GUI_local(window)
		windowWidth = 763
		windowHeight = 55
	else:
		gui = GUI_remote(window)
		windowWidth = 673
		windowHeight = 55

	positionRight = int(window.winfo_screenwidth()/2 - windowWidth/2)
	positionDown = int(window.winfo_screenheight()/2 - windowHeight/2)
	window.geometry("+{}+{}".format(positionRight, positionDown))
	window.lift()
	window.attributes('-topmost',True)
	window.after_idle(window.attributes,'-topmost',False)
	window.focus_force()
	window.mainloop()

	if cancelled == False:
		window = tkinter.Tk()
		message = show_message(window, positionRight, positionDown)
		thread = threading.Thread(target = threaded_function)
		thread.setDaemon(True)
		thread.start()
		message.mainloop()

		r = requests.post(get_rootMBR_url)
		response = r.json()
		c_query_x1 = float(response['content'][0])
		c_query_x2 = float(response['content'][0]) + float(response['content'][2])
		c_query_y1 = float(response['content'][1])
		c_query_y2 = float(response['content'][1]) + float(response['content'][3])
		transformer = Transformer.from_crs("epsg:3627", "epsg:3857")
		c_query_x1, c_query_y1 = transformer.transform(c_query_x1+150, c_query_y1+150)
		c_query_x2, c_query_y2 = transformer.transform(c_query_x2-150, c_query_y2-150)

		map = curdoc().select_one({'name' : "map"})
		c_x_diff = c_query_x2 - c_query_x1
		c_y_diff = c_query_y2 - c_query_y1
		x_diff = map.x_range.end - map.x_range.start
		y_diff = map.y_range.end - map.y_range.start

		if c_x_diff > c_y_diff:
			scale_f = x_diff/y_diff
			new_c_y_diff = c_x_diff/scale_f
			map.x_range.start = c_query_x1
			map.x_range.end = c_query_x2
			map.y_range.start = c_query_y1
			map.y_range.end = c_query_y1 + new_c_y_diff
			default_query_x1 = query_x1 = c_query_x1
			default_query_x2 = query_x2 = c_query_x2
			default_query_y1 = query_y1 = c_query_y1
			default_query_y2 = query_y2 = c_query_y1 + new_c_y_diff
			update_plot()
		else:
			scale_f = y_diff/x_diff
			new_c_x_diff = c_y_diff/scale_f
			map.x_range.start = c_query_x1
			map.x_range.end = c_query_x1 + new_c_x_diff
			map.y_range.start = c_query_y1
			map.y_range.end = c_query_y2
			default_query_x1 = query_x1 = c_query_x1
			default_query_x2 = query_x2 = c_query_x1 + new_c_x_diff
			default_query_y1 = query_y1 = c_query_y1
			default_query_y2 = query_y2 = c_query_y2
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
	    b.css_classes=['custom_button']
	buttons[0].css_classes=['custom_button_selected']
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
		scale_f = x_diff/y_diff
		new_c_y_diff = c_x_diff/scale_f
		map.x_range.start = c_query_x1
		map.x_range.end = c_query_x2
		map.y_range.start = c_query_y1
		map.y_range.end = c_query_y1 + new_c_y_diff
		query_x1 = c_query_x1
		query_x2 = c_query_x2
		query_y1 = c_query_y1
		query_y2 = c_query_y1 + new_c_y_diff
		update_plot()
	else:
		scale_f = y_diff/x_diff
		new_c_x_diff = c_y_diff/scale_f
		map.x_range.start = c_query_x1
		map.x_range.end = c_query_x1 + new_c_x_diff
		map.y_range.start = c_query_y1
		map.y_range.end = c_query_y2
		query_x1 = c_query_x1
		query_x2 = c_query_x1 + new_c_x_diff
		query_y1 = c_query_y1
		query_y2 = c_query_y2
		update_plot()
####################################################################################
####################################################################################


####################################################################################
########################## VISUALIZATION INITIALIZATION ############################
####################################################################################

# Create the map plot for the default scale
tile_provider = get_provider(STAMEN_TERRAIN_RETINA)
globe = x_range, y_range = ((initial_query_x1, initial_query_x2), (initial_query_y1, initial_query_y2))
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
ts_plot_height = 200

for i in list(range(0, max_k1)):
    source = ColumnDataSource(
        data=dict(counts=[],
                  lat=[],
                  lon=[],
                  size=[],
                  colors=[],
                  fill_alpha=[],
                  line_alpha=[],
		          geometries=[],
		          top=[],
		          bottom=[],
		          left=[],
		          right=[])
    )
    sourcesMap.append(source)
    c = p.quad(top="top", bottom="bottom", left="left", right="right", fill_color="colors", fill_alpha="fill_alpha", line_alpha="line_alpha", line_color="colors", line_width=4, source=source, name=str(i))
    glyph = Quad(fill_color="colors", fill_alpha="fill_alpha", line_alpha="line_alpha", line_color="colors", line_width=4)
    c.selection_glyph = glyph
    c.nonselection_glyph = glyph

    labels = LabelSet(x='lon', y='lat', text_font='helvetica', text='counts', text_font_size='20px', text_font_style='bold', text_align='center', text_baseline='middle', text_color='white', background_fill_color='colors', level='overlay', x_offset=-7, y_offset=-7, source=source, render_mode='canvas', name="label " + str(i))
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
    buttons.append(Button(label=str(i+1), css_classes=['custom_button'], name="button " + str(i), sizing_mode = 'stretch_height'))
    s.height = ts_plot_height
    s.sizing_mode = 'stretch_width'
    button_row = row(buttons[i])
    button_row.width = 44
    button_row.height = ts_plot_height - 20
    layout = row(button_row, s, name="ts_plot " + str(i))
    layout.sizing_mode = 'stretch_both'
    ts_plots.append(layout)
##################################################################
##################################################################

# Add all extra cuntionality buttons and define their callbacks
fetchButton = Button(label="Fetch", css_classes=['custom_button_fetch'])
fetchButton.on_event(ButtonClick, fecth_data)

resetButton = Button(label="Reset", css_classes=['custom_button_reset'])
resetButton.on_event(ButtonClick, reset_plot)

menuTSsummaries = [("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"), ("6", "6"), ("7", "7"), ("8", "8"), ("9", "9"), ("10", "10")]
noOfTSsummaries = Dropdown(css_classes=['custom_button'], menu=menuTSsummaries)
noOfTSsummaries.label = str(default_k1) + " bundles"
noOfTSsummaries.on_click(selected_k1)

selectDataset = [("Open Local", "Open Local"), ("Open Remote", "Open Remote")]
selectDatasetDropdown = Dropdown(label="Open Dataset", css_classes=['custom_button'], menu=selectDataset)
selectDatasetDropdown.on_click(selected_dataset)

menuScale = [("1:500", "Scale 1:500"), ("1:5000", "Scale 1:5000"), ("1:10000", "Scale 1:10000"), ("1:15000", "Scale 1:15000"), ("1:20000", "Scale 1:20000"), ("1:25000", "Scale 1:25000")]
scaleSelect = Dropdown(css_classes=['custom_button'], menu=menuScale)
scaleSelect.on_click(selected_scale)

callbacks = [createFunc(i) for i in range(max_k1)]
for i in range(max_k1):
	buttons[i].on_click(getCallback(callbacks, i))
buttons[0].css_classes = ['custom_button_selected']

# Add everything to layouts and to the final application
ts_plots = column(ts_plots)
ts_plots.sizing_mode = 'stretch_both'
func_buttons = row(fetchButton, resetButton, selectDatasetDropdown, noOfTSsummaries, scaleSelect)
func_buttons.sizing_mode = 'stretch_width'
lay1 = column(ts_plots, name="ts_plots")
lay2 = column(func_buttons, p, name="map_div")
lay1.sizing_mode = 'stretch_width'
lay2.sizing_mode = 'stretch_both'
curdoc().add_root(lay1)
curdoc().add_root(lay2)
curdoc().title = "spaTScope"