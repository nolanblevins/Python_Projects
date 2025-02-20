# init vars to track max score
max_score = 0
best_combination = (0, 0)

# time constraints in minutes
total_time = 90
time_per_essay = 10
time_per_short_answer = 2

# points per question
points_per_essay = 20
points_per_short_answer = 5

# brute force approach
for X in range(3, 11):  # essay questions is between 3 and 10
    for Y in range(10, 51):  # multiple choice is between 10 and 50
        # check if current combination meets constraint of time
        if (X * time_per_essay + Y * time_per_short_answer) <= total_time:
            # compute score
            current_score = X * points_per_essay + Y * points_per_short_answer
            # update max score if better
            if current_score > max_score:
                max_score = current_score
                best_combination = (X, Y)

print(best_combination, max_score)
