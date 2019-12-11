Policy Improvement using HCOPE and CMA-ES


The requirement of running this code:

- Python 3.7
- Already installed packages: cma, scipy



You can either run it in command line or suitable IDE. For command line, cd to the folder that contains this Python file, and run "python3 PolicyImprovement.py".



Abstract of code:

1. Load data (state feature amount, action amount, features length in function approximation, histories, etc.)
2. Split history data into 2 parts for candidate histories and safety histories.
3. Maximize HCOPE score using the BBO method of CMA-ES. Then we can get serval optimal policies. As described in function (487) in course notes.
4. Do the candidate test. As described in function (488). Those policies who can pass this test can be retained for the next step safety test.
5. Do the safety test for the candidate policies, as function (479). Those policies who can pass this test will be added to the result policies.
6. Store the result policies into separate .csv files.



Dec. 10, 2019