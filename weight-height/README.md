# Weight-Height Plot

  ## Introduction
  The dataset given in this task consists of columns of gender, height, and weight, and gender is male and female, and height and weight are float data. Since the given task needs to load the data and calculate the histogram, KDE, and Gaussian distridution, the csv file was loaded first. I used Python's csv module to load the dataset, and the csv method to read the data in the form of a dictionary. Load the data and store it in a list by dividing it into height all, weight all, height male, height female, weight male, and weight female. The following is a part of the code written to load data. Additionally, I used colab and colab’s library drive mount on google drive.

  ### 1. Histograms
  The histogram can process data sequentially in a non-parametric method, and has a good effect on visualization, making it easy to understand the data. It is possible to estimate the local density of the histogram, and it is very important to adjust the parameter value for smoothing. Therefore, it is necessary to select a parameter value that is neither too large nor too small for the actual data. However, the histogram is difficult to apply because the estimated density is discontinuous.
  
  ### 2. KDE(Kernel Density Estimation)
  Kernel Density Estimation, one of the non-parametric density estimation methods, is a method that improves problems such as discontinuities in histograms by using  a kernel function. First, the kernel function is defined as a non-negative function that is symmetric about the origin and has an integral value of 1, and Gaussian, Epanechnikov, and uniform functions are typical kernel functions. KDE is expressed by the following formula.

  <img width="566" alt="image" src="https://user-images.githubusercontent.com/49769190/136489335-0557b89b-b8c8-475a-b9ea-d4dd58645125.png">

  ### 3. MLE
  MLE is a method of selecting a candidate that maximizes the likelihood function (or log likelihood function) among a number of candidates that can be the parameter θ of the assumed probability distribution as an estimator of the parameter. Likelihood refers to the likelihood that the data obtained now come from the distribution. Here parameters were estimated for that data using μML and σ2 ML. To calculate the likelihood numerically, the likelihood contribution from each data sample can be calculated and multiplied. The reason for multiplying the height is that the extraction of each data is an independent event. As shown in the equation below, the combined probability density function of the entire sample set is called the likelihood function.
  
  <img width="566" alt="image" src="https://user-images.githubusercontent.com/49769190/136489192-c5e38c94-ccf3-4e2f-9668-f576290f9d8e.png">

Usually, the log-likelihood function is used as follows, using natural logarithms.

  <img width="566" alt="image" src="https://user-images.githubusercontent.com/49769190/136489262-96bf272c-4f35-4210-a227-bb9982f64a23.png">

The MLE mean and MLE variance were calculated from the given data sample, and the curve was plotted using the Gaussian distribution function. The MLE shows the trend of a sample of data in a simple shape and describes the optimal approximate distribution. The following equations were use.
 
  <img width="619" alt="image" src="https://user-images.githubusercontent.com/49769190/136489383-79af0d99-de58-433f-a198-0d9aebcf5437.png">
  
  
