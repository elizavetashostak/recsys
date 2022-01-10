# RecSys course (EPAM Advanced DS course)
This repository contains the code for solving problems from the [RecSys course v.2](https://kb.epam.com/display/EPMCBDCCDS/RecSys+course).

### Task 0 (Metrics)
Using materials from the topic, implement the following metrics yourself:

* RMSE
* HitRate@k
* MAP@k
* NDCG@k

Evaluation criteria:
* RMSE – 1  
The metric is implemented correctly
* HitRate@k – 2  
The metric is implemented correctly
* MAP@k – 3  
The metric is implemented correctly
* NDCG@k – 4  
The metric is implemented correctly  

Total - 10

### Task 1
Dataset - [Instacart](https://www.kaggle.com/c/instacart-market-basket-analysis/data) (use this data for all tasks below)

EDA (2 points):
* Investigation of tables and mapping between them
* Distribution of users/ products/ orders
* Popularity of products/categories
* Existence of dummy products

Metrics (1 point):
* MAP@k
* HitRate@k
* NDCG@k

Models (7 points):
* Most popular recommender (1 point)
* SVD based recommender (3 points)
* Neural Network recommender which has 2 embedding layers for user and item and calculates the score as the dot product. Fully-connected layers can be incorporated as well to improve the model's expressiveness. (3 points)

Optional models (6 points):
* Nearest Neighbor recommender utilizing SVD embeddings and annoy library for fast neighbors search (2 points).
* ALS recommender from the implicit library (2 points).
* Gradient boosting recommender utilizing SVD embedding (2 points).  

Total - 10 points + 6 extra points

### Task 2

#### Task 2.1 (11 max)
Dataset - [Instacart](https://www.kaggle.com/c/instacart-market-basket-analysis/data)

Implement learning to rank approaches
* implement point-wise - 2
* implement pair-wise - 4
* There are descriptions and visualization of results - 3
* There are descriptions of the reasons for choosing metrics - 2  

**Note!** There is a specific in the Instacart dataset. There are no queries and pages as usually are in traditional datasets for learning to rank. To implement learning to rank you need to have query-depended features. Working with Instacart data you could generate some features to catch user-product interaction and add it as query-depended features. (e.g. days since the user bought a product, count of views/purchase of a product by user)). Also, you could generate any kind of user features (treating the user as a query), even any kind of embeddings of users.

#### Task 2.2 (8 max)

Investigate Kaggle competition solutions 

[1](https://www.kaggle.com/c/santander-product-recommendation/discussion) or [2](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/discussion) or [3](https://www.kaggle.com/c/instacart-market-basket-analysis/discussion)

Use top solutions notebooks or discussions for analysis.

The task:
* take one of the suggested competitions
* understand competition
* find solutions in discussions or notebooks with ensembles
* describe these solutions in text or scheme format
* Create a list of ensembles realization - a type of ensemble and link to the discussions/notebooks that were analyzed 
* (optionally) add your comment about what would you add to the solution to improve it

**Note:** some of the discussions don't have code links, but for this task, it's Ok just to understand the idea of the solutions.

#### Task 2.3 (12 max)
Dataset - [Instacart](https://www.kaggle.com/c/instacart-market-basket-analysis/data)

Combine all learned models and rankers to an ensemble

Meta classifier
* create meta classifier - 3 
* importance of models analysis - 2
* hard cases extraction - 2
* There are descriptions and visualization of results - 3
* There are descriptions of the reasons for choosing metrics - 2

Total - 31 points

**(Optional)** Siamese NN 
* Implement Siamese neural networks applied to algorithm selection in recommender systems
* check the idea here
* implementation example
* loss: hinge loss or use one of the following losses with motivation

### Task 3 (Serving the model with Flask/Docker)

Workflow:
* Save a model from task 1/2 (e.g. as a pickle)
* Using Flask (or any other library), build a server, which has GET /predict resource with an adjustable number of predictions per user and returns JSON with recommended items for the user
* Example: GET http://localhost:5000/predict?user_id=12345&k=10 shall recommend (i.e. return JSON with) 10 items for user_id=12345
* Expected JSON keys: 'user_id' (number), 'items' (array of recommended items)
* Build a Docker image (using Dockerfile) that runs the server
* Tips (steps of the Dockerfile):
* Take the python base image,
* Copy requirements.txt
* Install Flask and libraries you need to run the model (in requirements.txt)
* Copy your server script and model
* Set up running the server (with CMD directive)
* Run the built docker image with exposing port you use in Flask

Evaluation criteria:
* Flask - 5
* GET request, mentioned above, yields JSON of the desired structure
* Docker - 5
* Flask server is wrapped into Docker image

Total - 10