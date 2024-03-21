# Employee attrition (rotation)
## Project description
### Project goal:
Reducing company attrition rate by implementing the ML solution for identyfing employees who are at risk of leaving the company
### Research and development
There are many implementations in the net related to customer churn or employee attrition - at the time of developing this solution, majority of implementations relied on static employee / customer files - snaphot at given date with target column stated whether the person left or stayed in the company.
However, there are certain drawbacks of similar approach:
1. No time indicator - when the employee / customer will leave?
2. Target variable shows correctly the positive target (left/churned), however those employee who stayed are labeled as 0 (stayed) at **given point of time**. We are not certain about their future (maybe they will churn the next day?)
3. Having dataset constructed like described above impacts also the inference process. We are trying to predict when someone will churn, but employees / customers in inference dataset can be the same as they were in training dataset (where they were trained with label 0).

### Solution
Due to mentioned reasons, final solution is built differently:
1. Reference paper https://www.esann.org/sites/default/files/proceedings/2021/ES2021-110.pdf for using multiple observations per employee / customer.
2. Using only terminated employees / customers as base for training data - no mix for training / prediction dataset.

#### Metrics
Precision and recall were mainly used for solution assessment and shapley values as explainers. 

## Guide
Folder structure of the project looks like this:
- r&d - experiments with time-to-event analysis (survival analysis) with different methods.
- v2 - final version of chosen solution - multiple observations per employee (sequence-based)

## Tech stack
python, pandas, scikit-survival, sklearn, statsmodels