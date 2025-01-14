This is the repository for The Remedy Project's Bureau of Prisons Administrative Remedy Data Dashboard. Code is written in Python with [Plotly Dash](https://dash.plotly.com/).

# Local Development
Clone the repository to your local machine, and install the contents of the `requirements.txt` file using pip or conda (recommended to do so in a virtual environment with python 3.10).

Inside the virtual environment, to start the web app, run (from the repo's base directory): 
```
python src/data_dashboard.py
```

Then navigate to [http://localhost:8051/](http://localhost:8051/) in your local browser.

## Development Tools
### Debugging
To access the [dev tools suite](https://dash.plotly.com/devtools), you will want to set `Debug=True` in the `"__main__"` block of the code in `src/data_dashboard.py`. Note that this will lead to an additional process being initialized that will result in additional memory consumption.

### Profiling
Profiling can be done on a Linux/Unix system by starting the app with this command:
```
PROFILER="True" python src/data_dashboard.py
```
or equivalently on Windows, in Anaconda Powershell Prompt:
```
$env:PROFILER="True"; python src/data_dashboard.py
```
This will output `.prof` files to a directory named `profiling` in your present working directory. If this folder does not exist, you will want to create it before attempting to profile the code.

Examination of the `.prof` files can be done using [snakeviz](https://jiffyclub.github.io/snakeviz), where detailed instructions can be found [here](https://community.plotly.com/t/performance-profiling-dash-apps-with-werkzeug/65199).
