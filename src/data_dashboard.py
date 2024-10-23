import pandas as pd
import numpy as np
from datetime import datetime
import os, time
import random

# import matplotlib.pyplot as plt

# import re
# import json

# import plotly.express as px
import plotly.graph_objects as go
# from dash import Dash, html, dash_table, dcc, Input, Output, State, \
#     callback, callback_context
from dash import dash_table
from dash_extensions.enrich import DashProxy, Output, Input, State, Serverside, html, dcc, \
    ServersideOutputTransform, callback_context, callback

rel_path2fsbackend = './file_system_backend'

session_key = random.randint(100000,999999)

regional_office_codes = ['MXR', 'NCR', 'NER', 'SCR', 'SER', 'WXR']
central_office_code = 'BOP'

default_timerange = ['2000-01-01', '2024-06-01'] #datetime.today().strftime('%Y-%m-%d')]

complaint_data_dtype_dict = {
    "CASENBR": "int32",
    "ITERLVL": "category",
    "CDFCLEVN": "category",
    "CDFCLRCV": "category",
    "CDOFCRCV": "category",
    "CDSTATUS": "category",
    "STATRSN1": "category",
    "STATRSN2": "category",
    "STATRSN3": "category",
    "STATRSN4": "category",
    "STATRSN5": "category",
    "CDSUB1PR": "category",
    "CDSUB1SC": "category",
    "sdtdue": "datetime64[ns]",
    "sdtstat": "datetime64[ns]",
    "sitdtrcv": "datetime64[ns]",
    "accept": "boolean",
    "reject": "boolean",
    "deny": "boolean",
    "grant": "boolean",
    "other": "boolean",
    "submit": "boolean",
    "filed": "boolean",
    "diffreg_filed": "boolean",
    "diffinst": "boolean",
    "closed": "boolean",
    "comptime": "Int16",
    "timely": "boolean",
    "diffreg_answer": "boolean",
    "overdue": "boolean",
    "untimely": "boolean",
    "resubmit": "boolean",
    "noinfres": "boolean",
    "attachmt": "boolean",
    "wronglvl": "boolean",
    "otherrej": "boolean",
    "cdsub1cb": "category",
}

used_fields = ['ITERLVL','CDFCLRCV','CDOFCRCV','CDSTATUS','sitdtrcv',
               'accept','reject','deny','grant','other','cdsub1cb']

# load the complaint filings data into a pandas DataFrame
cpt_df = pd.read_parquet('../data/complaint-filings-optimized.parquet')[used_fields]
# cpt_df = pd.read_parquet('https://drive.google.com/uc?export=download&id=1ST06IlcakkLsR-KNoXtop1ut9QbAiDdC',)
# _parquet_kwargs = {"engine": "pyarrow",
#                    "compression": "brotli",
#                    "index": False}
# cpt_df.astype(complaint_data_dtype_dict).to_parquet('../data/complaint-filings-optimized.parquet', **_parquet_kwargs)
# read_mem = cpt_df.memory_usage().sum() / 1024 ** 2
# print(read_mem)

# cpt_df[['sdtdue', 'sdtstat', 'sitdtrcv']] = cpt_df[['sdtdue', 'sdtstat', 'sitdtrcv']].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce',)

name_key_df = pd.read_csv('../data/facility-info.csv',)

subj_codes_df = pd.read_csv('https://drive.google.com/uc?export=download&id=1OQ8xLLF3hG3Dtd9C_LJpvngfsH3YO-5B')

subj_code_opts = [{'label':row['secondary_desc'], 'value':row['code']} for i, row in subj_codes_df.iterrows()]
subj_code_opts = sorted(subj_code_opts, key=lambda x: x['label'])
# subj_code_opts = [{'label': 'SELECT ALL', 'value': 'all'}] + subj_code_opts

status_dict = {'CLD': 'Denied', 
               'CLO': 'Closed (Other)', 
               'CLG':'Granted', 
               'ACC':'Accepted', 
               'REJ':'Rejected'}

trp_color = '#1e374f'

color_map_pie = {
    'Rejected':trp_color,
    'Denied':'#DD6E42',
    'Closed (Other)':'#9882AC',
    'Granted':'#FFDEC2'
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = DashProxy(__name__, external_stylesheets=external_stylesheets, transforms=[ServersideOutputTransform()])

server = app.server

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Span('Choose filing level:',
                          style={'fontWeight': 'bold', 'margin-right': '20px'}),
                dcc.Checklist(
                    id='filing-level',
                    options=[
                        {
                            'label': 'Facility (BP9)',
                            'value': 'F'
                        },
                        {
                            'label': 'Region (BP10)',
                            'value': 'R'
                        },
                        {
                            'label': 'Agency (BP11)',
                            'value': 'A'
                        },
                    ],
                    value=['F', 'R', 'A'],
                    inline=True,
                ),
                dcc.Store(data=['F'], id='filing-store')
            ],
            style={'width': '100%', 'margin-bottom': '40px'}),

            html.Div([
                html.Span('Track cases by:',
                          style={'fontWeight': 'bold', 'margin-right': '20px'}),
                dcc.Dropdown(
                    id='tracking-level',
                    options=[
                        {
                            'label': 'Institution of Origin',
                            'value': 'CDFCLRCV'
                        },
                        {
                            'label': 'Office Responsible for Outcome',
                            'value': 'CDOFCRCV'
                        },
                    ],
                    clearable=False,
                    value='CDOFCRCV' #'CDFCLRCV',#'CDOFCRCV',
                ),
            ],
            style={'width': '100%'}),
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-right': '20px'}),
        # html.Div(
        #     [
        #         'Filter by case subject:',
        #         dcc.Dropdown(
        #             id="type-dropdown",
        #             optionHeight=55,
        #             options=subj_code_opts,
        #             value=['all'],
        #             multi=True,
        #         ),
        #     ],
        #     style={'width': '33%', 'display': 'inline-block', 'vertical-align':'top'},
        # ),
        html.Div([
            # html.Div([
            #     html.Span('Filter cases by subject:', style={'fontWeight': 'bold', 'margin-right': '20px'}),
            #     html.Button('Select All', id='all-button-genre', className='all-button' ),
            #     html.Button('Select None', id='none-button-genre', className='none-button'),
            # ],
            #     className="multi-filter",
            #     style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'},
            # ),
            html.Div([
                # Left-aligned text
                html.Span('Filter cases by subject:',
                          style={'fontWeight': 'bold', 'margin-right': '20px'}),  # Label

                # Buttons pushed to the right
                html.Div([
                    html.Button('Select All', id='all-button-subj', className='all-button',
                                style={'margin-right': '10px', 'padding': '0 10px', 'height': '25px',
                                       'font-size':'12px','line-height':'8px', 'vertical-align':'middle'}),
                    html.Button('Select None', id='none-button-subj', className='none-button',
                                style={'padding': '0 10px', 'height': '25px',
                                       'font-size':'12px','line-height':'8px', 'vertical-align':'middle'}),
                ], style={'display': 'flex'}),  # Inline-flex for the buttons
            ], style={
                'display': 'flex',
                'justify-content': 'space-between',  # Push the buttons to the right
                'align-items': 'bottom',  # Vertically align both the text and buttons
                'margin-bottom': '10px'
            }),

            html.Div([
                dash_table.DataTable(
                    id='datatable-subj-filter',
                    columns=[
                        {"name": '', "id": 'label'}
                    ],
                    data=subj_code_opts, #table that I defined at start
                    fixed_rows={'headers': False},
                    filter_action='native',
                    row_selectable="multi",
                    selected_rows=list(range(len(subj_code_opts))), # not needed done below instead
                    virtualization=False,
                    page_action='none',
                    style_table={
                        'minHeight': '120px',
                        'maxHeight': '120px',
                        'overflowY': 'auto',
                    },
                    css=[
                        {
                            'selector': '.dash-cell div.dash-cell-value',
                            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;',
                        },
                        {
                            'selector': 'tr:first-child',
                            'rule':'''
                                    display: None;
                            '''
                        },
                    ],
                    filter_options={
                        'case': 'insensitive',
                        'placeholder_text': 'Search for specific case subjects...',
                    },
                    style_header={
                        'backgroundColor': trp_color,
                        'color': 'white',
                        'fontSize': '14px',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                    },
                    style_cell={
                        'whiteSpace': 'no-wrap',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'maxWidth': 0,
                        'textAlign': 'left',
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(232, 232, 232)'
                        }
                    ],
                    style_as_list_view=True,
                ),
                # html.Div(id='datatable-interactivity-container')
            ],
            ),
        ],
        className='individual-filter',
        style={'width': '50%', 'display': 'inline-block', 'vertical-align':'top'},
        ),
    ], style={'display': 'flex', 'width': '100%'}),

    html.Div([
        html.Hr(
            style={'width': '100%', 'padding':'0px',},
        )
    ], style={'width': '100%', 'padding':'0px',}),
    # html.Div([
    #     dcc.Graph(id='institution-map')
    # ], style={'width': '75%', 'display': 'inline-block'}),
    html.Div([
        html.Div(
            html.Div([
                dcc.Graph(id='institution-map', clear_on_unhover=True)
            ]),
            id='graph-container',
            style={'width': '50%', 'display': 'inline-block',  'padding':'0px',}
        ),

        html.Div(
            dcc.Graph(
                id='institution-pie',
                figure={
                    'layout': go.Layout(
                        margin=dict(l=10, r=10, t=10, b=10),  # Tight margins
                    )
                }
            ),
            style={'width': '50%', 'display': 'inline-block', 'padding':'0px'}),
        ], style={'height':'300px'},
    ),
    html.Div(
        dcc.Graph(
            id='case-cts',
            figure={
                'layout': go.Layout(
                    margin=dict(l=0, r=0, t=0, b=0),  # Tight margins
                )
            }
        ),
        style={'width': '100%', 'display': 'inline-block', 'padding':'0px'}),

    # html.Div(dcc.RangeSlider(
    #     cpt_df['sitdtrcv'].min(),
    #     cpt_df['sitdtrcv'].min(),
    #     step=1,
    #     id='crossfilter-year--slider',
    #     value=df['Year'].max(),
    #     marks={str(year): str(year) for year in df['Year'].unique()}
    # ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
    dcc.Store(id='time_range', data=default_timerange),
    dcc.Store(id='filtered_df'),
    dcc.Store(id='institution_filtered_df'),
    dcc.Store(id='inst_name'),
])

@app.callback(
    Output('filtered_df', 'data'),
    inputs = [
        Input('filing-level', 'value'),
        Input('datatable-subj-filter', "selected_rows"),
        Input('time_range', 'data'),
        State('datatable-subj-filter', "data"),
    ],
)
def filter_df(filingSelections, selected_subj_rows, time_range, subj_rows):
    # clear out the files in the storage directory
    current_time = time.time()
    for fname in os.listdir(rel_path2fsbackend):
        fcreation_time = os.path.getctime(f'{rel_path2fsbackend}/{fname}')
        f_age_minutes = (current_time - fcreation_time) / (60) # divide by seconds per [time unit you're interested in]
        if f_age_minutes >= 60.:
            os.unlink(f'{rel_path2fsbackend}/{fname}')

    selected_subj_code_list = [subj_rows[i]['value'] for i in selected_subj_rows]
    filter_mask = cpt_df['ITERLVL'].isin(filingSelections) & \
                  cpt_df['cdsub1cb'].isin(selected_subj_code_list) & \
                  (cpt_df['sitdtrcv'] > time_range[0]) & (cpt_df['sitdtrcv'] < time_range[1])

    return Serverside(cpt_df[filter_mask], key=f'filter_df_{session_key}')

@app.callback(
    output = [
        Output('institution_filtered_df', 'data'),
        Output('inst_name', 'data')
    ],
    inputs = [
        Input('institution-map', 'hoverData'),
        Input('institution-map', 'clickData'),
        Input('filtered_df', 'data'),
        Input('tracking-level', 'value'),
    ],
)
def inst_filter_df(hoverData,clickData,dff,trackingSelection,):
    info = clickData if clickData else hoverData

    if info is None:
        inst_name = 'All Institutions'
    else:
        inst_code = info['points'][0]['customdata'][3]
        dff = dff[dff[trackingSelection] == inst_code]
        inst_name = name_key_df[name_key_df['facility_code']==inst_code]['nice_name'].values[0]

    return Serverside(dff, key=f'inst_filter_df_{session_key}'), inst_name

@app.callback(
    output = Output('time_range', 'data'),
    inputs = [
        Input('case-cts', 'relayoutData'),
        State('time_range', 'data'),
    ],
)
def update_time_range(casects_relayout, time_range):
    if casects_relayout:
        time_range = casects_relayout.get(
            'xaxis.range',
            [
                casects_relayout.get('xaxis.range[0]', default_timerange[0]),
                casects_relayout.get('xaxis.range[1]', default_timerange[1])
            ]
        )

    return time_range

@app.callback(
    [
        Output('datatable-subj-filter', "selected_rows"),
    ],
    [
        Input('all-button-subj', 'n_clicks'),
        Input('none-button-subj', 'n_clicks'),
        State('datatable-subj-filter', "data"), #"derived_virtual_data"),
    ]
)
def select_all_subj(all_clicks, none_clicks, selected_rows):
    if selected_rows is None:
        return []
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
        return list(range(len(subj_code_opts)))
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'all-button-subj':
        return [i for i in range(len(selected_rows))]
    else:
        return []

@app.callback(
    Output('institution-map', 'clickData'),
    [Input('graph-container', 'n_clicks')]
)
def reset_clickData(n_clicks):
    return None


@app.callback(
    Output('filing-level', 'value'),
    Output('filing-store', 'data'),
    Input('filing-level', 'value'),
    State('filing-store', 'data')
)
def update_checklist(value, active):
    """
    Prevent checklist from having no filing levels selected.
    """
    if len(value) < 1:
        return active, active
    else:
        return value, value

@app.callback(
    Output('institution-map', 'figure'),
    inputs = [
        Input('filtered_df', 'data'),
        Input('tracking-level', 'value'),
        State('time_range', 'data'),
    ],
)
def update_map(dff, trackingSelection, time_range):
    time_start_str = datetime.strptime(time_range[0].split(' ')[0], '%Y-%m-%d').strftime('%m/%Y')
    time_end_str = datetime.strptime(time_range[1].split(' ')[0], '%Y-%m-%d').strftime('%m/%Y')

    summary_df = dff.groupby(trackingSelection, sort=False, observed=True).agg(
        total_cases=('CDSTATUS', 'size'),
        rejected_cases=('reject', 'sum'),
        denied_cases=('deny', 'sum'),
        granted_cases=('grant', 'sum'),
        closed_other_cases=('other', 'sum'),
        accepted_cases=('accept', 'sum')
    ).reset_index()

    
    summary_df['total_closed_cases'] = summary_df['rejected_cases'] + summary_df['denied_cases'] + summary_df['granted_cases'] + summary_df['closed_other_cases']
    summary_df['no_remedy_frac'] = 1 - (summary_df['granted_cases'] / summary_df['total_closed_cases'])
    summary_df = pd.merge(summary_df, name_key_df, left_on=trackingSelection, right_on='facility_code')
    summary_df = summary_df[summary_df['latitude'].notnull()]
    summary_df['hover_template'] = np.where(summary_df['pop_total'].notnull(), 
                                            "<b>%{hovertext}</b><br>" + 
                                                 "2024 Population: %{customdata[0]:,}<br>" +
                                                 f"Total cases ({time_start_str}-{time_end_str}): " + "%{customdata[1]:,}<br>" +
                                                 "Rejection/Denial Rate: %{customdata[2]:.1%}<br>" +
                                                 "<extra></extra>", 
                                            "<b>%{hovertext}</b><br>" + 
                                                 f"Total cases ({time_start_str}-{time_end_str}): " + "%{customdata[1]:,}<br>" +
                                                 "Rejection/Denial Rate: %{customdata[2]:.1%}<br>" +
                                                 "<extra></extra>"
                                           )

    # Split the data based on category
    region_mask = summary_df[trackingSelection].isin(regional_office_codes)
    central_mask = summary_df[trackingSelection].isin([central_office_code])
    dff_F = summary_df[~np.logical_or(region_mask, central_mask)]
    dff_R = summary_df[region_mask]
    dff_A = summary_df[central_mask]

    sizemax = 20
    casetotalmax = len(dff)/50 #np.max(summary_df['total_closed_cases'])/10

    # Create the mapbox figure with multiple traces
    fig = go.Figure()
    
    # # Trace for facility (Reds colorscale)
    # fig.add_trace(go.scattermap(
    #     lat=dff_F['lat_adj'],
    #     lon=dff_F['long_adj'],
    #     mode='markers',
    #     marker=go.scattermapbox.Marker(
    #         size=dff_F['total_closed_cases'],
    #         color=dff_F['no_remedy_frac'],
    #         colorscale='Greens',
    #         cmin=0.9,  # Set min value for color scaling
    #         cmax=1.0,  # Set max value for color scaling
    #         showscale=False,  # Show the colorscale for this trace
    #         # colorbar=dict(title='Greens', x=0.85)  # Position of colorbar
    #         # sizemax=20,
    #         sizeref=20/np.max(dff_F['total_closed_cases']),
            
    #     ),
    #     text=dff_F["facility_name"], 
    #     customdata=[dff_F['pop_total'], dff_F['total_closed_cases'], dff_F['no_remedy_frac'], dff_F['facility_code']],
    #     name='Facility'
    # ))
    for test_df, test_cscale, locality in zip([dff_F, dff_R, dff_A], ['Reds', 'Greens', 'Blues'], ['Facility', 'Regional Office', 'BOP Headquarters']):
        fig.add_trace(go.Scattermap(
            lat=test_df['lat_adj'],
            lon=test_df['long_adj'],
            mode='markers',
            marker=go.scattermap.Marker(
                size=test_df['total_closed_cases'],
                color=test_df['no_remedy_frac'],
                colorscale=test_cscale, # Use Reds or another color scale if necessary
                cmin=0.5,
                cmax=1.0,
                sizeref=(2 * casetotalmax)/(sizemax**2),
                sizemode='area',
                sizemin=2,
            ),
            name=locality,
            hoverinfo='text',
            hovertext=test_df['nice_name'],
            customdata=test_df[['pop_total', 'total_closed_cases', 'no_remedy_frac', 'facility_code']],
            hovertemplate=test_df['hover_template'],
        ))
    # fig.add_trace(go.Scattermap(
    #     lat=dff_F['lat_adj'],
    #     lon=dff_F['long_adj'],
    #     mode='markers',
    #     marker=go.scattermap.Marker(
    #         size=dff_F['total_closed_cases'],
    #         color=dff_F['no_remedy_frac'],
    #         colorscale='Reds', # Use Reds or another color scale if necessary
    #         cmin=0.9,
    #         cmax=1.0,
    #         sizeref=(2 * np.max(dff_F['total_closed_cases']))/(sizemax**2),
    #         sizemode='area',
    #         # sizemin=4,
    #     ),
    #     hoverinfo='text',
    #     hovertext=dff_F['nice_name'],
    #     customdata=dff_F[['pop_total', 'total_closed_cases', 'no_remedy_frac', 'facility_code']],
    #     hovertemplate=dff_F['hover_template'],
    # ))
    
    # Update layout with Mapbox style and settings
    fig.update_layout(
        map_style="basic",
        map_zoom=2.7,
        map_center={"lat": 38, "lon": -95},
        margin={"r":0,"t":0,"l":0,"b":0},
    )

    
    # fig = px.scatter_map(summary_df,
    #                      lat="lat_adj", lon="long_adj", size="total_closed_cases", color="no_remedy_frac", 
    #                      color_continuous_scale=summary_df['color_scale'],#'Reds', 
    #                      size_max=20, hover_name="facility_name", zoom=2.7, range_color=[0.9,1.0],
    #                      hover_data=['pop_total', 'total_closed_cases', 'no_remedy_frac', 'facility_code'],
    #                     )
    # fig.update_layout(
    #     coloraxis={
    #         'colorbar': {
    #             'title': 'Rejection / Denial<br>Rate',
    #             'tickformat':'.0%',
    #             # 'cmin': 0.9,   # Set minimum value
    #             # 'cmax': 1.0  # Set maximum value
    #         }
    #     },
    #     margin=dict(l=0, r=0, t=0, b=0)
    # )
    # fig.update_layout(coloraxis_showscale=False)
    # fig.update_traces(
    #     hovertemplate=summary_df['hover_template'],
    #     # "<b>%{hovertext}</b><br>" + 
    #     #               "2024 Population: %{customdata[0]}<br>" +
    #     #               "Total cases (2000-2007): %{customdata[1]}<br>" +
    #     #               "Rejection/Denial Rate: %{customdata[2]:.1%}<br>" +
    #     #               "<extra></extra>", 
    #     # customdata=name_key_df[['pop_total']],  # Pass customdata for hovertemplate to access
    #     # custom_data=[summary_df['pop_total'], summary_df['total_closed_cases'], summary_df['no_remedy_frac']],  # Pass customdata for hovertemplate to access
    #     hovertext=summary_df['nice_name']  # Preserve the facility name in bold
    # )
    
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        uirevision='constant',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0)',
        ),
        height=300,
    )

    return fig
    

@app.callback(
    Output('institution-pie', 'figure'),
    inputs = [
        Input('institution_filtered_df', 'data'),
        Input('inst_name', 'data')
    ],
)
def update_pie(dff, inst_name):
    counts_df = dff['CDSTATUS'].value_counts()
    counts_df = counts_df.drop('ACC', errors='ignore')
    counts_df = counts_df.reindex(['REJ', 'CLG','CLD', 'CLO',])

    labels = [status_dict[status] for status in counts_df.index]
    
    colors = [color_map_pie.get(label, 'gray') for label in labels]

    # fig = px.pie(counts_df, 
    #              values = counts_df.values,
    #              names=[status_dict[status] for status in counts_df.index], 
    #              title='Case Results',
    #              color=[status_dict[status] for status in counts_df.index],
    #              color_discrete_map={'Rejected':'#1e374f',
    #                              'Denied':'#DD6E42',
    #                              'Closed (Other)':'#9882AC',
    #                              'Granted':'#FFE8D4'},
    #              sort=False,
    #             )
    fig = go.Figure(data=[go.Pie(labels=[status_dict[status] for status in counts_df.index],
                                 values=counts_df.values)])
    fig.update_traces(hoverinfo='label+percent', 
                      textinfo='value', 
                      # text=[val for val in counts_df.values],
                      textfont_size=18, pull=[0,0.3,0,0], sort=False, rotation=270,
                      marker=dict(colors=colors, line=dict(color='#000000', width=1)))
    fig.update_layout(
        title=f'Administrative Remedy Outcomes<br>({inst_name})',
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=300,
    )
    # fig.update_layout(
    #     legend=dict(
    #         yanchor="top",
    #         y=0.99,
    #         xanchor="left",
    #         x=-0.21
    #     )
    # )

    return fig

@app.callback(
    Output('case-cts', 'figure'),
    inputs=[
        Input('institution_filtered_df', 'data'),
        Input('inst_name', 'data'),
        State('time_range', 'data'),
    ],
)
def update_case_counts(dff, inst_name, time_range):
    case_counts_df = dff.set_index('sitdtrcv').resample('W').size().reset_index(name='case_count')
    case_counts_df['monthly_rolling_avg'] = case_counts_df['case_count'].rolling(window=4).mean()
    case_counts_df['monthly_rolling_sum'] = case_counts_df['case_count'].rolling(window=4, min_periods=1).sum()


    fig = go.Figure()

    # Add the actual event counts to the plot
    # fig.add_trace(go.Scatter(x=case_counts_df['sitdtrcv'], y=case_counts_df['case_count'],
    #                          mode='lines', name='Filing Count', line=dict(color=color_map_pie.get('Rejected'))))

    # Add the rolling average line to the plot
    fig.add_trace(go.Scatter(x=case_counts_df['sitdtrcv'], y=case_counts_df['monthly_rolling_sum'],
                             mode='lines', name='2-Month Rolling Average',
                             line=dict(color=trp_color, width=2)))

    fig.update_layout(
        title=f"Rolling Monthly Administrative Remedy Filings ({(inst_name)})",
        # xaxis_title="Time",
        # yaxis_title="Weekly Filing Count",
        xaxis = dict(
            rangeslider = {'visible': True},
            range = time_range if time_range!=default_timerange else None, # keep time_range permanently to keep everything 2000-2024
            autorange = False if time_range!=default_timerange else True, # keep false permanently to keep everything 2000-2024
        ),
        margin={"t": 40, "b": 0},
        height=250,
    )

    return fig
    
if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
