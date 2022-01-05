# Network Control Theory for neuroimaging data
## Description
This package contains wrapper functions around the [network_control](https://github.com/BassettLab/control_package) package facilitating the computation of network control theory related measures for mulitple subjects.
### Requirements
`pip install network_control`
### Installation
The package is currently not pip-installable. You can however download the repository and use the functions by running the following at the beginning of your script:
```python 
import sys
sys.path.append('path/to/nict')
from nict.multi_subject import function_of_your_choice