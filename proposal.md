---
layout: default
title: Project Proposal
---

## Introduction
In today’s world, product reviews are essential to determine the quality of the service providers, and affect the sales by the business. However, there has been an escalation of fake product reviews that can mislead other customers in their choices and decisions. Repercussions of such fake product reviews can affect both the customer and the service providers. With the advent of online service providers and e-commerce platforms, it has become necessary to identify fake reviews and ensure the integrity of the reviewing process. In this project, we plan to use Yelp review dataset to classify genuine reviews. We aim to demonstrate the efficacy of review features as inputs to unsupervised and supervised learning models and leverage the information available on user interactions. 

## Methods
The project will analyze restaurant reviews on Yelp, using a variety of both supervised and unsupervised machine learning techniques. We plan to use supervised learning methods like Logistic Regression, Gradient Boosting Tree, CNN, and BERT, etc to compare the results, while the unsupervised learning methods will likely include, among others, K-Means and the Gaussian Mixture Model (GMM) to cluster the restaurants in various clusters. 

## Potential Results
Fake Reviews prove to be harmful in both scenarios, where users give a good restaurant a bad review or give a bad review to a good restaurant. To combat this problem we propose to build a solution that would weed out fake reviews. We are targeting the restaurants in the Chicago and New York region, where we would have certain features - Food Review, User Rating, Restaurant Rating and Restaurant Location on a broad level. Further we will drill more into the context of food reviews and figure out the relation between the features, that would help us in derive further features. Using all the features we would like to build a model that would point out - Fake Reviews, Fake Users, Fake Restaurants and Localities with most fake reviews. The model will be able to detect — particular users who put up only fake reviews, particular restaurants that have maximum fake reviews to improve the rating, particular restaurants which have fake bad reviews to lower competition. This would further help us in cleaning the reviews and ratings of the restaurants. We will use accuracy, recall, precision and F1 score to determine the results. Final result would be the model where inputting the restaurant, the reviewer, rating and the review, we would be able to identify where the review is fake or genuine and should it be removed from the website. 

## Discussion
Segmentation of the market by use of novel features is an important step towards designing a market strategy for any company. Our project would help food delivery and cataloging firms to analyse which Restaurants have fake reviews and as a result of which the potential customer is mislead into buying a product that is actually not good. Moreover, removal of the identified fake reviews and ratings would serve as a data reconciliation measure and would help in improving the prediction results of existing systems. 

## References
[1] Shebuti Rayana, and Leman Akoglu. "Collective opinion spam detection: Bridging review networks and metadata." Proceedings of the 21th acm sigkdd international conference on knowledge discovery and data mining. ACM, 2015

[2] Leman Akoglu, Rishi Chandy, and Christos Faloutsos. "Opinion fraud detection in online reviews by network effects." Seventh international AAAI conference on weblogs and social media. 2013.

[3] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. “BERT: pre-training of deep bidirectional transformers for language understanding.” CoRR, abs/1810.04805, 2018.
