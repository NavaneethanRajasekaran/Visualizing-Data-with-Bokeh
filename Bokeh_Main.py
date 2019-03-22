
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from os.path import dirname, join

from bokeh.plotting import figure
from bokeh.models import (CategoricalColorMapper, HoverTool,ColumnDataSource, Panel,FuncTickFormatter, 
                          SingleIntervalTicker, LinearAxis)
from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider, Tabs, CheckboxButtonGroup, 
                                  TableColumn, DataTable, Select)
from bokeh.layouts import column, row, WidgetBox
from bokeh.palettes import Category20_16
from bokeh.io import curdoc
from scipy.stats import gaussian_kde
from bokeh.sampledata.us_states import data as states
from itertools import chain


# In[2]:

flights = pd.read_csv(join(dirname(__file__), 'data', 'flights.csv'),
                      index_col=0)

# Formatted Flight Delay Data for map
map_data = pd.read_csv(join(dirname(__file__), 'data', 'flights_map.csv'),
                       header=[0,1], index_col=0)

flights = flights.dropna(subset=['arr_delay'])



# In[3]:


flights.head()


# In[4]:


#creating function for Histogram
def histogram(flights):
    def make_dataset(carrier_list, range_start = -60, range_end = 120, bin_width = 5):
        by_carrier = pd.DataFrame(columns=['proportion', 'left', 'right', 'f_proportion', 'f_interval',
                                           'name', 'color'])
        range_extent = range_end - range_start
        
        for i, carrier_name in enumerate(carrier_list):
            subset = flights[flights['name'] == carrier_name]
            arr_hist, edges = np.histogram(subset['arr_delay'],
                                           bins = int(range_extent / bin_width), 
                                           range = [range_start, range_end])
            arr_df = pd.DataFrame({'proportion': arr_hist / np.sum(arr_hist), 'left': edges[:-1], 'right': edges[1:] })
            arr_df['f_proportion'] = ['%0.5f' % proportion for proportion in arr_df['proportion']]
            arr_df['f_interval'] = ['%d to %d minutes' % (left, right) for left, right in zip(arr_df['left'], arr_df['right'])]
            arr_df['name'] = carrier_name
            arr_df['color'] = Category20_16[i]
            by_carrier = by_carrier.append(arr_df)
        by_carrier = by_carrier.sort_values(['name', 'left'])
        return ColumnDataSource(by_carrier)
   # function for style 
    def style(p):
        p.title.align = 'center'
        p.title.text_font_size = '20pt'
        p.title.text_font = 'serif'
        p.xaxis.axis_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_style = 'bold'
        p.yaxis.axis_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_style = 'bold'
        p.xaxis.major_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'
        return p
    #function for plot
    def make_plot(src):
        p = figure(plot_width = 700, plot_height = 700,title = 'Histogram of Arrival Delays by Airline',
                   x_axis_label = 'Delay (min)', y_axis_label = 'Proportion')
        p.quad(source = src, bottom = 0, top = 'proportion', left = 'left', right = 'right',
               color = 'color', fill_alpha = 0.7, hover_fill_color = 'color', legend = 'name',
               hover_fill_alpha = 1.0, line_color = 'black')
        hover = HoverTool(tooltips=[('Carrier', '@name'),('Delay', '@f_interval'),('Proportion', '@f_proportion')],
                          mode='vline')
        p.add_tools(hover)
        p = style(p)
        return p
    #function for active interaction
    def update(attr, old, new):
        carriers_to_plot = [carrier_selection.labels[i] for i in carrier_selection.active]
        new_src = make_dataset(carriers_to_plot,range_start = range_select.value[0],range_end = range_select.value[1],
                               bin_width = binwidth_select.value)
        src.data.update(new_src.data)
        
    available_carriers = list(set(flights['name']))
    available_carriers.sort()
    airline_colors = Category20_16
    airline_colors.sort()
    carrier_selection = CheckboxGroup(labels=available_carriers,active = [0, 1])
    carrier_selection.on_change('active', update)
    binwidth_select = Slider(start = 1, end = 30, 
                                 step = 1, value = 5,
                                 title = 'Bin Width (min)')
    binwidth_select.on_change('value', update)
    range_select = RangeSlider(start = -60, end = 180, value = (-60, 120),
                                   step = 5, title = 'Range of Delays (min)')
    range_select.on_change('value', update)
    initial_carriers = [carrier_selection.labels[i] for i in carrier_selection.active]
    src = make_dataset(initial_carriers,range_start = range_select.value[0],range_end = range_select.value[1],
                           bin_width = binwidth_select.value)
    p = make_plot(src)
    controls = WidgetBox(carrier_selection, binwidth_select, range_select)
    layout = row(controls, p)
    tab = Panel(child=layout, title = 'Histogram')
    return tab


# In[5]:


#Density Distribution
def density(flights):
    def make_dataset(carrier_list, range_start, range_end, bandwidth):
        xs = []
        ys = []
        colors = []
        labels = []
        for i, carrier in enumerate(carrier_list):
            subset = flights[flights['name'] == carrier]
            subset = subset[subset['arr_delay'].between(range_start,range_end)]
            #gaussian_kde to esimate probability Density Function
            kde = gaussian_kde(subset['arr_delay'], bw_method=bandwidth)
            x = np.linspace(range_start, range_end, 100)
            y = kde.pdf(x)
            xs.append(list(x))
            ys.append(list(y))
            colors.append(airline_colors[i])
            labels.append(carrier)
            new_src = ColumnDataSource(data={'x': xs, 'y': ys,'color': colors, 'label': labels})
        return new_src
   
    def make_plot(src):
        p = figure(plot_width = 700, plot_height = 700,title = 'Density Plot of Arrival Delays by Airline',
                   x_axis_label = 'Delay (min)', y_axis_label = 'Density')
        p.multi_line('x', 'y', color = 'color', legend = 'label',line_width = 3,source = src)
        hover = HoverTool(tooltips=[('Carrier', '@label'),('Delay', '$x'),('Density', '$y')],line_policy = 'next')
        p.add_tools(hover)
        p = style(p)
        return p

    def update(attr, old, new):
        carriers_to_plot = [carrier_selection.labels[i] for i in carrier_selection.active]
        if bandwidth_choose.active == []:
            bandwidth = None
        else:
                bandwidth = bandwidth_select.value
                new_src = make_dataset(carriers_to_plot,range_start = range_select.value[0],range_end = range_select.value[1],
                                       bandwidth = bandwidth)
        src.data.update(new_src.data)

    def style(p):
        p.title.align = 'center'
        p.title.text_font_size = '20pt'
        p.title.text_font = 'serif'
        p.xaxis.axis_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_style = 'bold'
        p.yaxis.axis_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_style = 'bold'
        p.xaxis.major_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'
        return p

    available_carriers = list(set(flights['name']))
    available_carriers.sort()

    airline_colors = Category20_16
    airline_colors.sort()

    carrier_selection = CheckboxGroup(labels=available_carriers,active = [0, 1])
    carrier_selection.on_change('active', update)

    range_select = RangeSlider(start = -60, end = 180, value = (-60, 120),step = 5, title = 'Range of Delays (min)')
    range_select.on_change('value', update)
    initial_carriers = [carrier_selection.labels[i] for i in carrier_selection.active]
    #bandwidth of kernel 
    bandwidth_select = Slider(start = 0.1, end = 5,step = 0.1, value = 0.5,title = 'Bandwidth for Density Plot')
    bandwidth_select.on_change('value', update)
    bandwidth_choose = CheckboxButtonGroup(
        labels=['Choose Bandwidth (Else Auto)'], active = [])
    bandwidth_choose.on_change('active', update)
    #Density Data Source
    src = make_dataset(initial_carriers,range_start = range_select.value[0],range_end = range_select.value[1],
                       bandwidth = bandwidth_select.value) 

    p = make_plot(src)

    p = style(p)

    controls = WidgetBox(carrier_selection, range_select,bandwidth_select, bandwidth_choose)

    layout = row(controls, p)

    tab = Panel(child=layout, title = 'Density Plot')
    return tab


# In[6]:


#Map Chart
def maps(map_data, states):
    def make_dataset(carrier_list):
        subset = map_data[map_data['carrier']['Unnamed: 3_level_1'].isin(carrier_list)]
        color_dict = {carrier: color for carrier, color in zip(available_carriers, airline_colors)}
        flight_x = []
        flight_y = []
        colors = []
        carriers = []
        counts = []
        mean_delays = []
        min_delays = []
        max_delays = []
        dest_loc = []
        origin_x_loc = []
        origin_y_loc = []
        dest_x_loc = []
        dest_y_loc = []
        origins = []
        dests = []
        distances = []
        for carrier in carrier_list:
            sub_carrier = subset[subset['carrier']['Unnamed: 3_level_1'] == carrier]
            for _, row in sub_carrier.iterrows():
                colors.append(color_dict[carrier])
                carriers.append(carrier)
                origins.append(row['origin']['Unnamed: 1_level_1'])
                dests.append(row['dest']['Unnamed: 2_level_1'])
                origin_x_loc.append(row['start_long']['Unnamed: 20_level_1'])
                origin_y_loc.append(row['start_lati']['Unnamed: 21_level_1'])
                dest_x_loc.append(row['end_long']['Unnamed: 22_level_1'])
                dest_y_loc.append(row['end_lati']['Unnamed: 23_level_1'])
                flight_x.append([row['start_long']['Unnamed: 20_level_1'],row['end_long']['Unnamed: 22_level_1']])
                flight_y.append([row['start_lati']['Unnamed: 21_level_1'],row['end_lati']['Unnamed: 23_level_1']])
                counts.append(row['arr_delay']['count'])
                mean_delays.append(row['arr_delay']['mean'])
                min_delays.append(row['arr_delay']['min'])
                max_delays.append(row['arr_delay']['max'])
                distances.append(row['distance']['mean'])
        new_src = ColumnDataSource(data = {'carrier': carriers, 'flight_x': flight_x, 'flight_y': flight_y,'origin_x_loc': origin_x_loc, 'origin_y_loc': origin_y_loc,
                                           'dest_x_loc': dest_x_loc, 'dest_y_loc': dest_y_loc,
                                           'color': colors, 'count': counts, 'mean_delay': mean_delays,
                                           'origin': origins, 'dest': dests, 'distance': distances,
                                           'min_delay': min_delays, 'max_delay': max_delays})
        return new_src
    
    def make_plot(src, xs, ys):
        p = figure(plot_width = 1100, plot_height = 700, title = 'Map of 2013 Flight Delays Departing NYC')
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.grid.visible = False
        patches_glyph = p.patches(xs, ys, fill_alpha=0.2, fill_color = 'lightgray',line_color="#884444", line_width=2, line_alpha=0.8)
        lines_glyph = p.multi_line('flight_x', 'flight_y', color = 'color', line_width = 2, 
                                   line_alpha = 0.8, hover_line_alpha = 1.0, hover_line_color = 'color',
                                   legend = 'carrier', source = src)
        squares_glyph = p.square('origin_x_loc', 'origin_y_loc', color = 'color', size = 10, source = src,legend = 'carrier')
        circles_glyph = p.circle('dest_x_loc', 'dest_y_loc', color = 'color', size = 10, source = src,legend = 'carrier')
        p.renderers.append(patches_glyph)
        p.renderers.append(lines_glyph)
        p.renderers.append(squares_glyph)
        p.renderers.append(circles_glyph)
        hover_line = HoverTool(tooltips=[('Airline', '@carrier'),
                                         ('Number of Flights', '@count'),('Average Delay', '@mean_delay{0.0}'),
                                         ('Max Delay', '@max_delay{0.0}'),
                                         ('Min Delay', '@min_delay{0.0}')],
                               line_policy = 'next',renderers = [lines_glyph])
        hover_circle = HoverTool(tooltips=[('Origin', '@origin'),('Dest', '@dest'),('Distance (miles)', '@distance')],
                                 renderers = [circles_glyph])
        p.legend.location = (10, 50)
        p.add_tools(hover_line)
        p.add_tools(hover_circle)
        p = style(p) 
        return p
    def style(p):
        p.title.align = 'center'
        p.title.text_font_size = '20pt'
        p.title.text_font = 'serif'
        p.xaxis.axis_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_style = 'bold'
        p.yaxis.axis_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_style = 'bold'
        p.xaxis.major_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'
        return p
    def update(attr, old, new):
        carrier_list = [carrier_selection.labels[i] for i in carrier_selection.active]
        new_src = make_dataset(carrier_list)
        src.data.update(new_src.data)
    available_carriers = list(set(map_data['carrier']['Unnamed: 3_level_1']))
    available_carriers.sort()
    airline_colors = Category20_16
    airline_colors.sort()
    if 'HI' in states: del states['HI']
    if 'AK' in states: del states['AK']
    xs = [states[state]['lons'] for state in states]
    ys = [states[state]['lats'] for state in states]
    carrier_selection = CheckboxGroup(labels=available_carriers, active = [0, 1])
    carrier_selection.on_change('active', update)
    initial_carriers = [carrier_selection.labels[i] for i in carrier_selection.active]
    src = make_dataset(initial_carriers)
    p = make_plot(src, xs, ys)
    layout = row(carrier_selection, p)
    tab = Panel(child = layout, title = 'Flight Map')

    return tab


# In[7]:


#summary stats of table 
def table(flights):
    carrier_stats = flights.groupby('name')['arr_delay'].describe()
    carrier_stats = carrier_stats.reset_index().rename(columns={'name': 'airline', 'count': 'flights', '50%':'median'})
    carrier_stats['mean'] = carrier_stats['mean'].round(2)
    carrier_src = ColumnDataSource(carrier_stats)
    table_columns = [TableColumn(field='airline', title='Airline'),
                     TableColumn(field='flights', title='Number of Flights'),
                     TableColumn(field='min', title='Min Delay'),
                     TableColumn(field='mean', title='Mean Delay'),
                     TableColumn(field='median', title='Median Delay'),
                     TableColumn(field='max', title='Max Delay')]
    carrier_table = DataTable(source=carrier_src,columns=table_columns, width=1000)
    tab = Panel(child = carrier_table, title = 'Summary Table')
    return tab


# In[8]:


tab1 = histogram(flights)
tab2 = density(flights)
tab3 = maps(map_data,states)
tab4 = table(flights)
tabs = Tabs(tabs=[tab1,tab2,tab3,tab4])


# In[9]:


curdoc().add_root(tabs)

