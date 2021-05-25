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
model =    [('X', [])
			]

varEquations = [
			    'X = logistic(-2,1) if choice([0,1]) else logistic(2,1)'
		        ]


				
