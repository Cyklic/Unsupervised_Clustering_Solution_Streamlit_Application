# Mall Customer Application
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://leonard-umoru-unsupervised-clustering-solution.streamlit.app/)

This application aims to segment mall customers based on a number of factors and shopping habits by leveraging machine learning predictions.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter details such as age, annual income, and spending score.
- Real-time prediction of loan eligibility based on the trained model.
- Accessible via Streamlit Community Cloud.

## Dataset
Malls are often indulged in the race to increase their customers and making sales. To achieve this task machine learning is being applied by many malls already.

It is amazing to realize the fact that how machine learning can aid in such ambitions. The shopping malls make use of their customers’ data and develop ML models to target the right audience for right product marketing.

Goal: Build an unsupervised clustering model to segment customers into correct groups.

Specifics:

Machine Learning task: Clustering model
Target variable: N/A
Input variables: Refer to data dictionary below
Success Criteria: Cannot be validated beforehand
Data Dictionary:
CustomerID: Unique ID assigned to the customer
Gender: Gender of the customer
Age: Age of the customer
Income: Annual Income of the customers in 1000 dollars
Spending_Score: Score assigned between 1-100 by the mall based on customer' spending behavior

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

## Model
The predictive model applies scaling numerical features. The classification model used is KMeans for segmentation.

## Future Enhancements
* Adding support for multiple datasets.
* Adding visualizations to better represent user input and model predictions.

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit_eligibility_application.git
   cd credit_eligibility_application

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:
   ```bash
   streamlit run app.py

#### Thank you for using the Mall Customer Segmentation Application! Feel free to share your feedback.
