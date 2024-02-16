# Personalized Recommendation Proposal

## 1. The Problem

We want to build a recommendation engine to suggest content to the users
that they are more likely to engage with. The content needs to be
suggested based on the content previously engaged with, and content
similar users engage in. Along with that the user's environment (recent
attitude) should be gauged, and suggestions should be incorporated
accordingly.

Although a user might not manually follow another user, the system
should automatically recognise the people who’s content a user enjoys
seeing. Let's develop a technique to measure this and call it an
“Implicit follow” feature. We shall discuss about the implicit follow
feature in the next section (2.3)

One of plans mentioned in the AI-ML infrastructure document was, *“As we
move further, we would want to consider 2nd and 3rd order profiles as
well.*For eg, content profiles of people a user follows, content
profiles of people engaging with the same poll as the user.” This
feature is the same as the user based collaborative filtering. This
model can be incorporated

## 2. Solution Discussions 

### 2.1 Hybrid Approach 

To build a recommendation engine we will use a hybrid version of content
based filtering & collaborative filtering.

#### 2.1.1 Content Based Filtering : 

In the content-based recommendation system, we retrieve the poll's
category and employ Natural Language Processing (NLP) to extract
additional information such as the question's subject, tone, and other
related features. By analyzing this feature set, we can discern a user's
preferences for engaging with polls. For instance, if a user frequently
engages with polls featuring whimsical questions about girlfriends, the
attribute associated with "girlfriends" will be prominently reflected in
the user's profile. This user profile, individually established for each
user, serves as a representation of their preferences.

The recommendation process involves merging both the content and user
profile, facilitating a more personalized and tailored recommendation.

<img src="media/image4.png" style="width:1.86458in;height:1.22112in" />

Each user will have 3 different different groups of characteristics that
the user likes. Each set of characteristic posts will ensure a certain
type of engagement. Example :

<table style="width:100%;">
<colgroup>
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 26%" />
<col style="width: 17%" />
<col style="width: 8%" />
<col style="width: 17%" />
</colgroup>
<thead>
<tr class="header">
<th></th>
<th></th>
<th><strong>characteristic 1</strong></th>
<th><strong>characteristic 2</strong></th>
<th><strong>...</strong></th>
<th><strong>characteristic X</strong></th>
</tr>
<tr class="odd">
<th><em>user1</em></th>
<th><em>overall</em></th>
<th>w1*(0.3) + w2*(0.7) + w3*(0.1)</th>
<th>–do–</th>
<th></th>
<th>–do–</th>
</tr>
<tr class="header">
<th></th>
<th><em>answered</em></th>
<th>0.3</th>
<th>0</th>
<th>...</th>
<th>0.7</th>
</tr>
<tr class="odd">
<th></th>
<th><em>shares</em></th>
<th>0.7</th>
<th>0</th>
<th>...</th>
<th>0</th>
</tr>
<tr class="header">
<th></th>
<th><em>comment + like</em></th>
<th>0.1</th>
<th>0.5</th>
<th>...</th>
<th>0</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

Where characteristic1 might be *subject_girlfirend*, characteristic2
might be *subject_Political*, and characteristic3 might be
*would_you_rather_tone*. Along with that we can have an overall
engagement metric for a user, based on weights determined by the goals
of the organization.

#### 2.1.2 User Based - Collaborative Filtering : 

This profile will also include other features, such as the user's
demographic and background information. At Level 1, user-user similarity
can be measured using demographic information.

Following this, a user-item matrix will be created with users as rows
and items as columns. This matrix will store the engagement of a
particular user with all items. Similarity can be computed between the
user and all other users to identify those with similar tastes.
Therefore, if polls that the user has not yet engaged with have been
engaged with by users similar to them, those polls can be assigned a
higher propensity for engagement. Even for collaborative filtering we
can product

<img src="media/image3.png" style="width:5.08854in;height:2.65685in" />

###  

### 2.2 Session Attitude : 

Depending on the user's behavior, setting, temperament, the engagement
of the user with the polls will not be the same. For instance, during
traveling (home-office) there is a high possibility that the user is
only interested in reading comments written on polls. Thus, each session
that the user has can be categorized differently. Some hypothesized
ideas are shown below :

<table>
<colgroup>
<col style="width: 24%" />
<col style="width: 38%" />
<col style="width: 36%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Attitude</strong></th>
<th><strong>Measure</strong></th>
<th><strong>Type of Recommendation</strong></th>
</tr>
<tr class="odd">
<th><em>Passive session</em></th>
<th>Time spent on each poll can be used to determine the session
type</th>
<th>Show posts that user expands and reads more often</th>
</tr>
<tr class="header">
<th><em>High active session</em></th>
<th>Comments and Polls Answered</th>
<th>Show user posts with a high propensity to answer</th>
</tr>
<tr class="odd">
<th><em>..</em></th>
<th>..</th>
<th>..</th>
</tr>
<tr class="header">
<th><em>..</em></th>
<th>..</th>
<th>..</th>
</tr>
<tr class="odd">
<th><em>Frequent Interruption</em></th>
<th>Multiple short sessions with high frequency</th>
<th><p>Show the user content with a high propensity to share</p>
<p>(User is engaged elsewhere too)</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

The user's interaction in the first few minutes / first round of
suggestion, can be used to categorize the users ‘Attitude’ using Machine
Learning Techniques.

### 2.3 Implicit Follow : 

Let's say there is a user A who has done a set of engagement in time
‘T’. And if user A’s engagements are similar to another user B’s
engagements in a time period ‘∂T’ after ‘T’. Then we can say that B is
implicitly following A. An algorithm can be made to find out implicit
followers of an account. This data-point can be augmented with the
features data point.

## 3. Model Selection and Approach for POC

Based on literature, collaborative learning performs better than Content
Based Learning. For a single model focused POC, the recommendation would
be to select collaborative filtering.

#### Different Approach to Collaborative Filtering : 

Other than directly finding similarity between users, based on from the
user_item engagement matrix, the following methods can be used :

- Clustering algorithms

- Matrix Factorization based algorithm

- Deep Learning methods

1)  **Clustering Algorithms:** They normally use simple clustering
    > Algorithms like K-Nearest Neighbours to find the K closest
    > neighbors or embeddings given a user or an item embedding based on
    > the similarity metrics used.

2)  **Matrix Factorization based algorithms:** The user-item interaction
    > matrix can also be factorized into two smaller matrices, and these
    > two matrices can also be used to generate back the interaction
    > matrix. So, we generate the factor matrices as feature matrices
    > for users and items. These feature matrices serve as embeddings
    > for each user and item. To create the feature matrices we need
    > dimensional reduction.  
    >   
    > The dimensionality reduction can be done by several methods:

    1.  SVD: Singular Value Decomposition

    2.  PMF: Probability Matrix Factorization

    3.  NMF: Non-Negative Matrix Factorization

<img src="media/image2.png" style="width:4.72805in;height:1.57293in" />

3)  **Deep Learning Methods :**

> SOTA (state of the art) Deep learning methods employ item features
> along with user_item interaction. While providing weights to each
> item. Since all items do not classify users taste, it is able to put
> weights to those items. Elaborated below.

#### Why is deep learning better than Other approaches ?

Primary difference: Matrix factorization assumes user interactions (1
for positive feedback, 0 for no interaction) directly represent user
preferences, even though this may not accurately reflect user likes or
dislikes. In contrast, deep learning approaches consider the nuances of
user engagement, acknowledging that an interaction (1) doesn't
necessarily indicate liking the content, and 0 may simply mean
non-engagement or missing data, highlighting the need to address the
challenge of negative feedback.

####  

#### Why to prefer Matrix Based Collaborative Filtering for POC ? 

The approach suggested is to use the user_item matrix to find user-user
similarity. Rather than selection bias, there is an rejection bias, on
why not to choose Deep Learning for POC -

- Data Scarcity: Deep learning models need substantial data, making them
  > less suitable for scenarios with limited user-item interaction data.

- Interpretability: Deep learning models are often considered black
  > boxes, lacking transparency in how recommendations are generated.

- Cold Start Problem: Deep learning models struggle with new users or
  > items with limited interaction history, where traditional methods
  > may perform better.

- Complexity: Neural networks can be resource-intensive requiring
  > extended training time, and careful hyperparameter tuning.

Thus higher preference should be given to Matrix based collaborative
learning approaches.

#### What are the primary and the secondary goals?

**Primary Goals :**

*<u>Demonstrate Improved User Engagement:</u>* Show that the
recommendation engine enhances user engagement by delivering more
relevant and personalized content.

> Monitor - metrics such as increased user interactions, longer session
> durations, and higher content consumption. Pre-post analysis of user
> retention & increased frequency
>
> Activities - Collect user feedback on the quality and relevance of
> recommendations. Evaluate if users find the recommended content
> valuable and engaging.

- 

**Secondary Goals :**

*<u>Validate Algorithm Suitability:</u>* Assess the effectiveness of the
chosen recommendation algorithms for the social media context.

Monitor - Experiment A/B test the POC approach based on the available
data.

##  

## 4. Model Deployment 

- Create Pipelines : Make pipeline for data to be preprocessed when
  > called by an API. Expected the recommendation engine to update every
  > time the user refreshes the app. Along with that set appropriate
  > cool down, to prevent refresh until a certain time period.

- Containerization : The use of containerization tools like Docker to
  > package the model and its dependencies, facilitating consistent
  > deployment across different environments.

- Orchestration : Tools like Kubernetes for managing and automating the
  > deployment, scaling, and operation of containerized applications.

- Monitoring and Logging: Systems for monitoring the performance of
  > deployed models, logging metrics, and generating alerts for any
  > anomalies or issues.

- Feedback Loop: Mechanisms to collect feedback from the deployed models
  > to continuously improve and update them based on real-world
  > performance and changing data patterns.

## 5. Maintenance | Feedback | Improvement

The table provides a comprehensive overview of the metrics relevant for
assessing the performance and impact of a collaborative filtering
recommendation engine in the context of social media post
recommendations.

<table>
<colgroup>
<col style="width: 58%" />
<col style="width: 41%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Business Perspective Metrics</strong></th>
<th><strong>Model Perspective Metrics</strong></th>
</tr>
<tr class="odd">
<th>1. Click-Through Rate (CTR)</th>
<th>1. Precision at K</th>
</tr>
<tr class="header">
<th>2. Likes, Shares, Comments</th>
<th>2. Recall at K</th>
</tr>
<tr class="odd">
<th>3. User Retention Rate</th>
<th>3. F1 Score at K</th>
</tr>
<tr class="header">
<th>4. Conversion Rate</th>
<th>4. Novelty Score</th>
</tr>
<tr class="odd">
<th>5. Time Spent on Recommended Content</th>
<th>5. Diversity Index</th>
</tr>
<tr class="header">
<th></th>
<th>6. Serendipity Score</th>
</tr>
<tr class="odd">
<th></th>
<th>7. User Satisfaction Surveys</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

The following formula can be used to measure the Novelty, Diversity &
Serendipity Metrics of the Model.

## <img src="media/image1.png" style="width:5.67188in;height:1.27254in" />

Along with model performance parameter, some key system performance
parameters include -

- Response Time: The response time of a recommendation system is the
  > duration it takes to provide recommendations in response to a user
  > query or request. This metric is critical as it directly influences
  > the system's responsiveness, impacting the overall user experience
  > by determining how quickly users receive relevant suggestions.

- Throughput: Throughput quantifies the system's capacity to generate
  > recommendations within a specific timeframe. It reflects the
  > system's ability to handle a substantial volume of user interactions
  > efficiently. Higher throughput ensures that the recommendation
  > engine can deliver recommendations promptly, even during periods of
  > increased user activity.

- Resource Utilization: Resource utilization measures how efficiently
  > the recommendation system utilizes computational resources,
  > including CPU, memory, and storage, during the recommendation
  > processes. Monitoring resource utilization is essential for
  > optimizing performance, maintaining stability, and preventing
  > resource-related bottlenecks.

- Latency: Latency refers to the delay or lag experienced by users when
  > receiving recommendations. It directly impacts the real-time
  > responsiveness of the system and influences user satisfaction. Lower
  > latency contributes to a more seamless and instantaneous user
  > experience.

Cohort analysis of these parameters can help us understand the next
steps to be taken for model improvement and can act as a feedback loop.

##  

## 6. Big Data Issues when scaling

Scaling a recommendation engine for big data introduces challenges,
including strain on computational resources, potentially impacting
real-time responsiveness. Continuous monitoring of response time,
throughput, and system latency is essential.

<img src="media/image5.png" style="width:6.5in;height:3.56944in" />

In handling millions of users and terabytes of data, scalability is
crucial. The architecture must support horizontal scaling for both model
training and recommendation inference. Techniques like Load Balancing
are vital for effective computational workload distribution.

Parallel processing is key to efficient handling of extensive data
volumes. Utilizing techniques like Data Parallelism, Model Parallelism,
Task Parallelism, Pipeline Parallelism, GPU Acceleration, and
Distributed Computing enhances efficiency, allowing concurrent
processing.

Designing a recommendation system to seamlessly handle large-scale data
involves strategic infrastructure implementation. Cloud-based solutions,
distributed computing, and parallel processing algorithms are critical.
Regular performance monitoring and optimization, including potential
adjustments to the collaborative filtering algorithm, ensure optimal
functionality as the system scales up.
