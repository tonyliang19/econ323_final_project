# Econ 323

- NOT TOO SIMPLE
- Research (ML methods level like CPSC330, Visualizations (matpolotlib, altair) )
	+ Complexity higher --> Maybe could do feature importance viz
	+ Prepocessing could use scikit-learn ?
	+ Could use multiple metrics:
		* Inference: AIC/BIC, mallow CP, ...
		* Prediction: RMSE, MAE 
- LASSO
- Decision Tree
- Manual train/compute model by hand
	+ Regression use $(X^{T}X)^{-1}X^{T}y$ like matrix multiplication with numpy?
	+ Manual split train/test 
## Deadlines
- 5 PM Thurs, April 6 **First Meetup** 
- 5 PM Thurs, April 13 **Second Meetup**
- Monday, April 24 11:59pm **Submission Deadline**

## Outline
	Intro
		| add section to briefly explain 
	EDA + Visualization
		| Outlier
		| Missing Value
		| Distribution of samples (each variable)
	Modelling
		| Linear
		| Ridge 
		| LASSO
		| Decision Tree
		| Random Forest
	Conclusion
	References （Intext and source for some model theory...)

## Methods

- Take off one Ridge and Decision Tree. Keep this simple and just have fewer models to train and explain
- Try not to have explanations that heavily math or stat based. (Ideally in the context of original variables)

## Outline

1. Intro
	+ Dataset description
2. Methods
	+ EDA
		* Basic info
		* Heat Map
		* Histogram of some chosen var
		* Boxplot distribution
	+ Model Fitting
		* Overview
			+ Common split data
			+ Fit all to validate/preview which might perform better
			+ Use common metric
		* OLS
		* LASSO
		* RF
	+ Model Selection
		* `Choose one from above of model fitting`
		* `Feature Selection of selected`
		* `optional optimize parameter`
4. Conclusion
	+ `final model performance (words)`
	+ `model show outcome with plot/table`
5. Discussion
	+ `Limitations of final model`
	+ `How to improve and what to consider`
	+ `What the model means in the original problem.`
6. References