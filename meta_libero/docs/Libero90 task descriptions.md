



# First 10 Libero 90 tasks
Using lr 2.5e-03, 5 steps


Task 0 `close the top drawer of the cabinet`
- Often tries to pick up the plate and put it into the drawer (similar to a seen task)
- TTT all steps: [20%, 20%, 18%]
- No TTT: [6%, 10%, 6%]

Task 1 `close the top drawer of the cabinet and put the bowl on top of it`
- 0% -> 0% with TTT
- Often picks up the plate and put it into the drawer, just sometimes closes it and doesn't have tiime to pick the bowl

Task 2 `put the black bowl in the top drawer of the cabinet`
- 90% but the behavior is suboptimal, gets stuck on the drawer (can we redirect it somehow with TTT?)
- Even though the TTT has around 45%, when it succeeds it looks like it removes this "gets stuck" behavior
    - Would be useful to compare the average num steps (when succeeding) as additional metric

Task 3 `put the butter at the back in the top drawer of the cabinet and close it`
- 0% -> 0%
- Normally, it doesn't pick up the right object, starts putting other stuff in the drawer
- But actually he knows how to pick up the right object, as happens sometimes (even if it doesn't succeed in closing the top drawer)
    - For instance, look at `/cluster/home/anmari/meta_vlas/meta_libero/results/ttt/lora/libero_90_task_3/lr2.50e-5_freq20_steps0_k6_seed4/videos/rollout_ttt_task3_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_ep23_failure.mp4`

Task 4 `put the butter at the front in the top drawer of the cabinet and close it`
Same problem task 3
- 0% -> 0%

Task 5 `put the chocolate pudding at the front in the top drawer of the cabinet and close it`
Same probkem task 3 and 4, moreover always starts from plate
- 0% -> 0%

Task 6 `open the bottom drawer of the cabinet`
- 0%, always fails because the cabinet has 3 drawers here (probably was trained on one with 2) and consistently opens the middle one
- First 150 steps only TTT: [10%, 30%, 14%]
- All steps TTT: [0%, 6%, 4%]
- No TTT: [0%, 0%, 0%]

Task 7 `open the top drawer of the cabinet`
- only one success in 3 seeds, always fails because the cabinet has 3 drawers here (probably was trained on one with 2) and consistently opens the middle one. (similarly to task 7).
- No TTT: [0%, 2%, 0%]
- First 150 steps only TTT: [8%, 2%, 6%]
- All steps TTT: [0%, 0%, 0%]

Task 8 `open the top drawer of the cabinet and put the bowl in it`
- Now it can sometimes open the top drawer (also here 3 drawers, so weird wrt to task 7) but gets stuck there
- Sometimes cannot even open the top drawer
- It has 3 successes in total (it knows how to solve the task for sure).
- No TTT: [0%, 4%, 2%]
- First 150 steps only TTT: [2%, 0%, 2%]
- All steps TTT: [2%, 0%, 0%]

Task 9 `put the black bowl on the plate`
- 90% success rate, when it fails is because it fails to position exactly, also tries to adjust it

Task 10 `put the black bowl on top of the cabinet`
- 90% success rate, probably has seen similar task during training, not too far away
