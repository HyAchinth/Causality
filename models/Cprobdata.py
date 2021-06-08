# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Initialize any variables here
t = 0

# Describe the test
testDescript = 'random distrubution'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')  
model =    ['X','Y',
			]

varEquations = [
			    'X = logistic(0,1)',
				'Y = X**2 + normal(0,0.1)'
		        ]


				
