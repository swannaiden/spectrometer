import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc 
import dash_html_components as html
from dash.dependencies import Input, Output, ALL, State, MATCH, ALLSMALLER


import plotly.graph_objects as go
import numpy as np

from cv2 import cv2
from flask import Flask, Response

#somebody on stackoverflow said to use Pillow, bc PIL is dead idk --christian
from PIL import Image

import math
import webbrowser
import os

import pandas as pd
import urllib

#do we need threading?

"""
Simple module that monkey patches pkg_resources.get_distribution used by dash
to determine the version of Flask-Compress which is not available with a
flask_compress.__version__ attribute. Known to work with dash==1.16.3 and
PyInstaller==3.6.
"""


#pyinstaller -c -F --add-data "assets/my.css;assets" --hidden-import "flask-compress" --clean postthanksgivinglayout.py
'''
import pkg_resources

IS_FROZEN = hasattr(sys, '_MEIPASS')

# backup true function
_true_get_distribution = pkg_resources.get_distribution
# create small placeholder for the dash call
# _flask_compress_version = parse_version(get_distribution("flask-compress").version)
_Dist = namedtuple('_Dist', ['version'])

def _get_distribution(dist):
    if IS_FROZEN and dist == 'flask-compress':
        return _Dist('1.8.0')
    else:
        return _true_get_distribution(dist)

# monkey patch the function so it can work once frozen and pkg_resources is of
# no help
pkg_resources.get_distribution = _get_distribution
'''


# Global Variables
'''
outputFrame = None
source = 0
server = Flask(__name__)
cam = VideoStream(src=source).start()
lock = threading.Lock()
'''
cameraCrop = [[0,160], [280, 180]]
exposure = 0
contrast = 0
cameraID = 0
spectrumWave = np.linspace(0,1000, cameraCrop[1][0] - cameraCrop[0][0])
spectrum = []
ref_spectrum = []
is_ref = False
is_abs = False
server = Flask(__name__)
#what is this Image function?
graph_background = Image.open('assets/spectrumCrop.jpg')


# initialize app stuff
# stylesheet, same one I use on my website (https://cjs3.cc)

external_stylesheets = ['my.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)


# VideoCamera object
class VideoCamera(object):
    def __init__(self):
        global cameraID
        self.video = cv2.VideoCapture(cameraID)
    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()


        cv2.normalize(image, image, (0+exposure + contrast), (255 + exposure - contrast), cv2.NORM_MINMAX)

        global cameraCrop
        spec = normalise(getSpectrum(image[cameraCrop[0][1] : cameraCrop[1][1], cameraCrop[0][0]: cameraCrop[1][0]]))
        global spectrum
        spectrum = spec

        
       
        image = cv2.rectangle(image, (cameraCrop[0][0], cameraCrop[0][1]), (cameraCrop[1][0], cameraCrop[1][1]), (0,0, 255), 3)
        ret, jpeg = cv2.imencode('.png', image[:, :])


        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Creating the figure
x = np.arange(700)
fig = go.Figure(data=go.Scatter(x=x, y=x*2))

# app layout/the html involved. Can use markup too for most of it if needed
app.layout = html.Div([
    # This div is the top half
    html.Div([
        # This div is the left column
        html.Div([
            # The s^3 logo, make sure to import it into the assets folder
            html.Img(src=app.get_asset_url("s3 logo.png"), alt="s^3 logo", style={"width":"70%"}),
            # Camera Settings heading
            html.H3("Camera Settings"),
            # Exposure label and sliders
            html.Div(
                [html.Label("Exposure", style={"vertical-align":"top"}),
                dcc.Input(id = "Exposure", type="range", min="-200", max="200", value="0", style={"width":"60%","float":"right"})], style={"top-padding":"5px"}
            ),
            html.Br(),
            # Contrast label and slider
            html.Div(
                [html.Label("Contrast", style={"vertical-align":"top"}),
                dcc.Input(id = "Contrast", type="range", min="-100", max="100", value="0", style={"width":"60%","float":"right"})], style={"top-padding":"5px"}
            ),
            html.Br(),
            # Saturation label and slider
            html.Div(
                [html.Label("Saturation", style={"vertical-align":"top"}),
                dcc.Input(id = "Saturation", type="range", min="1", max="100", value="50", style={"width":"60%","float":"right"})], style={"top-padding":"5px"}
            ),
        ], style={"background":"#f3faf0", "padding":"15px"}, className="grid-item"),
    

        # This div is the center column
        html.Div([
            # This div contains the camera
            html.Div(id = "camera", children=[
                # The video feed
                html.Img(src = "/video_feed", alt = 'Video Feed', width="100%", id = "video_frame")]
            ),
            # Camera input textbox
            dcc.Input(id="camera_name", placeholder="Camera ID", type="text",style={"width":"60%"}),
            # Refresh page button
            html.A(html.Button('Change Camera'),href='/', style={"margin-left":"5%", "float":"right"}),
            html.Br(),
        # The dropdown menu
            html.Div(
                # "How does this work?" text
                children = [html.P("How does this work?", style={"font-size":"0.5em","display":"inline","margin-bottom":"0px", "padding":"0px"}),
                # Text in the dropdown menu
                html.Div(
                    children= [
                        html.P("Type 0 for built-in camera input", style={"font-size":".8em"}), 
                        html.P("Type 1 for USB camera input", style={"font-size":".8em"}), 
                        html.P("Type an IP address for IP (phone) webcam", style={"font-size":".8em"}), 
                        html.P("Example IP adress: http://123.456.7.890:1234", style={"font-size":".75em"})
                    ],
                className="dropdown-content")], 
            className = "dropdown")]
        , style={"background":"#f9faf0", "padding":"15px"}, className="grid-item"),

        #The right column
        html.Div([
            html.H3("Set Input Line"),
            # Set both endpoints option
            html.H4("Set Endpoints"),
            # Point 1
            html.P("Point 1",style={"margin-bottom":"0px", "padding":"0px"}),
                #x1 input
                dcc.Input(id="x1", placeholder="x1", type="text",style={"width":"40%"}),
                #I should almost certianly use padding to space the text boxes instead of this tactical two space approach, but also like it works so
                html.P(",  ", style={"display":"inline", "vertical-align":"bottom", "font-size":"1.5em"}),
                #y1 input
                dcc.Input(id="y1", placeholder="y1", type="text",style={"width":"40%"}),
            # Point 2
            html.P("Point 2", style={"margin-bottom":"0px", "padding":"0px"}),
                #x2 input
                dcc.Input(id="x2", placeholder="x2", type="text",style={"width":"40%"}),
                html.P(",  ", style={"display":"inline", "vertical-align":"bottom", "font-size":"1.5em"}),
                #y2 input
                dcc.Input(id="y2", placeholder="y2", type="text",style={"width":"40%"}),
            
            # Slope intercept option
            html.H4("Slope Intercept"),
                #x intercept option
                dcc.Input(id="x-int", placeholder="x Intercept", type="text",style={"width":"43%"}),
                html.P("  or  ", style={"display":"inline", "font-size":"1em"}),
                #y intercept option
                dcc.Input(id="y-int", placeholder="y Intercept", type="text",style={"width":"43%"}),
                html.Label("Slope", style={"vertical-align":"top"}),
                # Slope slider
                dcc.Input(type="range", min="1", max="100", value="50", style={"width":"70%","float":"right"})
        ],style={"background":"#f0f7fa", "padding":"15px"}, className="grid-item"),
    ], className="grid-container"),
    
    # The bottom Sections
    html.Div([
        # The graph box
        html.Div([
            # The graph itself
            dcc.Graph(id="graph", figure=fig, style={"margin-bottom":"10px"}), 
            dcc.Interval(id = 'interval', interval = 300, n_intervals = 0),
            
            # Callibration
            html.Button("G", disabled=True, className="green", style={"vertical-align":"middle", "margin-bottom":"10px"}),
            # The green slider
            dcc.Input(type="range", min="0", max="1000", value="0", style={"width":"90%", "float":"right"}),
            html.Br(),
            html.Button("B", disabled=True, className="blue",style={"vertical-align":"middle"}),
            # The blue slider
            dcc.Input(type="range", min="0", max="1000", value="0", style={"width":"90%", "float":"right"}),
            html.Br(),
        ], style={"background":"#faf4f0", "padding":"15px"}, className="bgrid-item"),

        # The graph options box
        html.Div([
            html.H3("Graph Options"),
            #Buttons Section
            html.H4("Graph Display"),
            # Intensity button
            html.Button("Intensity", id="idInten", n_clicks=0, style={"margin-bottom":"10px", "margin-right":"5px"}),
            html.Br(),
            # Absorbance button
            html.Button("Absorbance", id="idAbsrob", n_clicks=0, style={"margin-bottom":"10px", "margin-right":"5px"}),
            html.Br(),
            # Calibration button
            html.Button("Reference", id="idCalib", n_clicks=0, style={"margin-bottom":"10px", "margin-right":"5px"}),

            # Name and save options
            html.Div([
                html.H4("Name & Save"),
                # Name spectrum input
                dcc.Input(id="spectrum_name", placeholder="Name Spectrum", type="text", style={"width":"100%"}),
                html.Br(),
                # Record dpectrum button
                #html.Button("Record Spectrum", id="record", n_clicks=0)
                html.A(html.Button(
                'Download Data'),
                id='download-link',
                download="rawdata.csv",
                href="",
                target="_blank")

            ], style={"vertial-align":"bottom"}),
        ], style={"background":"#faf0fa", "padding":"15px"}, className="bgrid-item"),
    ], className="bgrid-container"),

    # Aiden's Placeholders
    # I am clearly missing something in how to use html
    html.P(id='placeholder2', n_clicks = 0),
    html.P(id='placeholder3', n_clicks = 0),
    html.P(id='placeholder', n_clicks = 0),
    html.P(id='placeholder4', n_clicks = 0),
    html.P(id='placeholder5', n_clicks = 0),
    html.P(id='placeholder6', n_clicks = 0),

    # The help dropdown
    html.Details(children=[
        # The title of the help dropdown
        html.Summary("Help"),
        # The inside of the menu.
        html.P("This is where we could put a quick how to for users that may be confused.")
    ], style={"width":"99.5%"})
])
 
'''app callbacks'''

@app.callback(
    Output('download-link', 'download'),
    [Input('spectrum_name', 'value')])
def update_download_name(value):
    print(value)
    if(value == None):
        return "rawdata.csv"
    
    if(value == ''):
        return "rawdata.csv"

    return value + '.csv'
    

@app.callback(
    Output('download-link', 'href'),
    [Input('spectrum_name', 'value')])
def update_download_link(value):

    print(value)
    if(value == None):
        return ''
    
    if(value == ''):
        return ''

    output = {'Wavelength': spectrumWave, 'Intensity': spectrum}
    dff = pd.DataFrame(output)
    csv_string = dff.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string


@app.callback(
    Output('placeholder6', 'n_clicks'),
    [Input("Exposure", 'value'), Input("Contrast", 'value'), Input("Saturation", 'value')])
def updateCamSettings(exp, con, sat):

    global exposure
    global contrast

    exposure = int(float(exp))
    contrast = -1*int(float(con))

    return 1

# should lock image size here. This would allow us to impose an absolute max
@app.callback(
    Output('placeholder5', 'n_clicks'),
    [Input("x1", "value"), Input("y1", "value"), Input("x2", "value"), Input("y2", "value")])
def updatePoints(x1, y1, x2, y2):

    global cameraCrop
    if(x1 != None and x1 != ''):
        if(x1.isdigit()):
            if(int(x1) < cameraCrop[1][0]):
                cameraCrop[0][0] = int(x1)

    if(x2 != None and x2 != ''):
        if(x2.isdigit()):
            if(int(x2) > cameraCrop[0][0]):
                cameraCrop[1][0] = int(x2)

    if(y1 != None and x2 != ''):
        if(y1.isdigit()):
            if(int(y1) < cameraCrop[1][1]):
                cameraCrop[0][1] = int(y1)

    if(y2 != None and y2 != ''):
        if(y2.isdigit()):
            if(int(y2) > cameraCrop[0][1]):
                cameraCrop[1][1] = int(y2)

    return 1


@app.callback(
    Output('placeholder', 'n_clicks'),
    [Input("camera_name", "value")])
def updateCamera(value):

    if(value == None):
        return 0
    
    if(value == ''):
        return 0

    global cameraID
    if(value.isdigit()):
        cameraID = int(value)
        return 1
    else:
        cameraID = value
        return 1

@app.callback(
    Output('placeholder2', 'n_clicks'),
    [Input("idCalib", "n_clicks")])
def toggleRef(n):

    if(n == 0):
        return 0
    
    global is_ref

    #is_ref = n % 2
    is_ref = 1
    global ref_spectrum
    if(is_ref):
        ref_spectrum = spectrum.copy()
    
    return 1


@app.callback(
    Output('placeholder3', 'n_clicks'),
    [Input("idInten", "n_clicks")])
def toggleIntensity(n):
    global is_ref
    global is_abs
    is_ref = is_abs = 0

    return 1

@app.callback(
    Output('placeholder4', 'n_clicks'),
    [Input("idAbsrob", "n_clicks")])
def toggleAbs(n):

    if(n == 0):
        return 0

    global is_ref
    global is_abs
    is_ref = 0
    is_abs = 1

    global ref_spectrum
    ref_spectrum = spectrum.copy()

    return 1


@app.callback(Output('graph', 'figure'),
              [Input('interval', 'n_intervals')])
def update_graph(n):
    
    layout = {
    "margin" : {
        "l" : 20,
        "r" : 20,
        "t" : 20,
        "b" : 20
    },
    # Commenting out width and height to make graph fit the app margins
    #"width": 1044, 
    "xaxis": {
        "type": "linear", 
        "range": [0, 1000], 
        "title": "Wavelength (nm)", 
        "autorange": False
    }, 
    "yaxis": {
        "type": "linear", 
        "range": [0, 1], 
        "title": "Absorbance", 
        "autorange": True
    }, 
    #"height": 527, 
    "autosize": False
    }

    fig = go.Figure(layout = layout)

    #specToGraph = []
    if(is_ref):
        specToGraph = calcRef(ref_spectrum, spectrum)
    elif(is_abs):
        specToGraph = calcAbs(ref_spectrum, spectrum)
    else:
        specToGraph = spectrum

    trace1 = {
    "uid": "5bdd99", 
    "name": "Absorbance", 
    "type": "scatter", 
    #this is where the calibration happens
    "x": np.linspace(0, 1000, len(spectrum)),
    "y": specToGraph #random.randint(100, size=(28))#["56.7", "74.6", "95", "97.9", "100", "98", "94.6", "91.3", "87", "83", "77.5", "57.9", "34.9", "22.6", "18", "18.4", "19.6", "20.1", "24.1", "31.5", "38.5", "42", "46.6", "57.1", "62.3", "60.3", "72", "86.4", "87.6"]
    }

    #need to calibrate the graph background as well
    global graph_background
    fig.add_layout_image(
            dict(
                source=graph_background,
                xref="x",
                yref="y",
                x=200,
                y=2,
                sizex=600,
                sizey=4,
                sizing="stretch",
                opacity=0.5,
                layer="below"
                )
    )
    fig.add_trace(trace1)
    fig.update_layout(template="plotly_white")
    
    return fig


'''Spectrum functions'''
def getSpectrum(image):                                     
    '''From a numpy image array generate the uncalibrated spectrum'''          
    #
    # Reading the image                                            
    #image = cv2.imread(filename)                                  
    #
    # Preparing the variables                                      
    imageR = []                                                    
    imageG = []                                                    
    imageB = []                                                    
    imgWidth = len(image[0])                                       
    imgHeight = len(image)                                         
    
    # Preparing the RGB arrays                                     
    for i in range(imgWidth):                                      
        imageR.append(int(image[0][i][0]))                              
        imageG.append(int(image[0][i][1]))                              
        imageB.append(int(image[0][i][2]))                           
    #
    # Columns summatory                                            
    for i in range(imgHeight):                                     
        for j in range(imgWidth):                                  
            imageR[j]=imageR[j]+image[i][j][0]                     
            imageG[j]=imageG[j]+image[i][j][1]                     
            imageB[j]=imageB[j]+image[i][j][2]                     
    #
    # Calculating the mean for every RGB column                    
    for i in range(imgWidth):                                      
        imageR[i]=imageR[i]/imgHeight                              
        imageG[i]=imageG[i]/imgHeight                              
        imageB[i]=imageB[i]/imgHeight                              
    #
    # Merging the RGB channels by addition                         
    spectrum = []                                                  
    for i in range(imgWidth):                                      
        spectrum.append((imageR[i]+imageG[i]+imageB[i])/3)         
    #
    # returning the results of the operation                       
    return spectrum                                                

def normalise(spectrumIn):
    
    spectrumOut = []
    
    maxPoint = max(spectrumIn)
    
    #invalid input error
    for value in spectrumIn:
        spectrumOut.append(value/maxPoint)
    
    return spectrumOut


def calcAbs(reference, sample):
    # Calculate transmittance and absorbance spectrum
    
    transmittance = []
    absorbance = []
    
    for i in range(len(reference)):
        if sample[i] == 0: # This 'if' part is to aqvoid math error due to 0/number
            transmittance.append(0)
            absorbance.append(0) # Conceptually wrong, if sample > reference, artificious data distortion has happened
        else:
            transmittance.append(sample[i]/reference[i])
            absorbance.append(-math.log(transmittance[i],10)/5)

    return absorbance



def calcRef(reference, sample):

    ref = [0] * len(reference)
    #print('calculated ref')
    for i in range(len(reference)):
        ref[i] = (reference[i] - sample[i]) 

    return ref




if __name__ == '__main__':

    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:8050/')
    
    app.run_server(debug=True)

