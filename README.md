# linear_regression
This is my first work on machine learning
The initial step was linear regression with one variable Linear regression is a statistical approach for modelling relationship between a dependent variable with a given set of independent variables.In order to provide a basic understanding of linear regression, we start with the most basic version of linear regression, i.e. Simple linear regression.Simple linear regression is an approach for predicting a response using a single feature.

It is assumed that the two variables are linearly related. Hence, we try to find a linear function that predicts the response value(y) as accurately as possible as a function of the feature or independent variable(x).

For generality, we define:

x as feature vector, i.e x = [x_1, x_2, …., x_n],

y as response vector, i.e y = [y_1, y_2, …., y_n]


The equation of regression line is represented as:

h(x_i) = \beta _0 + \beta_1x_i

Here,
    h(x_i) represents the predicted response value for ith observation.
    b_0 and b_1 are regression coefficients and represent y-intercept and slope of regression line respectively.
    
To create our model, we must “learn” or estimate the values of regression coefficients b_0 and b_1. And once we’ve estimated these coefficients, we can use the model to predict responses!


y_i = \beta_0 + \beta_1x_i + \varepsilon_i = h(x_i) + \varepsilon_i \Rightarrow \varepsilon_i = y_i -h(x_i)

Here, e_i is residual error in ith observation.
So, our aim is to minimize the total residual error.

We define the squared error or cost function, J as:
J(\beta_0,\beta_1)= \frac{1}{2n} \sum_{i=1}^{n} \varepsilon_i^{2}


and our task is to find the value of b_0 and b_1 for which J(b_0,b_1) is minimum!

Without going into the mathematical details, we present the result here:

\beta_1 = \frac{SS_{xy}}{SS_{xx}}

\beta_0 = \bar{y} - \beta_1\bar{x}

where SS_xy is the sum of cross-deviations of y and x:
SS_{xy} = \sum_{i=1}^{n} (x_i-\bar{x})(y_i-\bar{y}) = \sum_{i=1}^{n} y_ix_i - n\bar{x}\bar{y}

and SS_xx is the sum of squared deviations of x:
SS_{xx} = \sum_{i=1}^{n} (x_i-\bar{x})^2 = \sum_{i=1}^{n}x_i^2 - n(\bar{x})^2 

In this, I have not used an algorithm rather just the mathematics and python libraries
