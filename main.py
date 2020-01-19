from review import Review
from user import User
from item import Item
import numpy as np
# 1. Data manipulation has been skipped due to weird assignments
#
# 2. Learning
# a. This dataset fits better with collaborative filtering. 
#    For content based filtering we would need more data about the movies to make it good.
#	 With Collaborative filtering we can assume that people have simililar tastes in movies
#
#
# b. Subtract the movie and user means from the training data
#    Rating(User, Movie) = Rating(User, Movie) 
#     - (1 / Total number of observed ratings for Movie) * ???? 
#     - (1 / total number of observed ratings for User) * ?????) 
#     + (1 / total number of observed movie-user pairs) * ?????) 
#
# c. Construct a matrix factorization CF model (a.k.a "Funk-SVD") for this training data. Use between 10 - 50 factors
#    
#
# d. Does matrix factorization seem as the better choice of CF technique for this data? Why / Why not?
#    For our case the matrix factorization seems very slow, as it has to compute a lot of computations, and when the matrix get as large
#    as they are, then it takes way more time.
#

def main():
    review_list = readReviews('u1.base')
    user_list = readUsers()
    item_list = readItems()

    Recommender(user_list, review_list, item_list)



def Recommender(user_list, review_list, item_list):
    number_of_iterations = 10000
    number_of_factors = 10
    number_of_users = len(user_list)
    number_of_items = len(item_list)
    
    user_review_matrix = PopulateUserReviewMatrix(user_list, review_list, item_list)

    # Indsætter tilfældige værdier mellem 0 og 1
    user_factor_matrix = np.random.rand(number_of_users, number_of_factors)
    factor_item_matrix = np.random.rand(number_of_factors, number_of_items)
    number_of_rows = number_of_users
    number_of_cols = number_of_items
    for i in range(0, number_of_iterations):
        predicted_ratings = DotMatrices(number_of_rows, number_of_cols, number_of_factors, user_factor_matrix, factor_item_matrix)
        

        alpha_value = 0.001
        beta_value = 0.001
        for r in range(0, number_of_rows):
            for c in range(0, number_of_cols):
                if user_review_matrix[r + 1][c + 1] > 0:
                    error = user_review_matrix[r + 1][c + 1] - predicted_ratings[r][c]
                    for f in range(0, number_of_factors):
                        user_factor_matrix[r][f] = user_factor_matrix[r][f] + alpha_value * (error * factor_item_matrix[f][c] - beta_value * user_factor_matrix[r][f])
                        factor_item_matrix[f][c] = factor_item_matrix[f][c] + alpha_value * (error * user_factor_matrix[r][f] - beta_value * factor_item_matrix[f][c])
        
        predicted_ratings = DotMatrices(number_of_rows, number_of_cols, number_of_factors, user_factor_matrix, factor_item_matrix)
        
        square_error = 0

        for r in range(0, number_of_rows):
            for c in range(0, number_of_cols):
                if(user_review_matrix[r + 1][c + 1] > 0):
                    square_error = square_error + np.power(user_review_matrix[r + 1][c + 1] - predicted_ratings[r][c], 2)
                    for f in range(0, number_of_factors):
                        square_error = square_error + beta_value * (np.power(user_factor_matrix[r][f], 2) + np.power(factor_item_matrix[f][c], 2))  
    
        print(str(i) + ': ' + str(square_error))
    for r in range(0, number_of_rows):
        for c in range(0, number_of_cols):
            if user_review_matrix[r + 1][c + 1] == 0:
                user_review_matrix[r + 1][c + 1] = predicted_ratings[r][c]
            else:
                user_review_matrix[r + 1][c + 1] = 0


    print('hejsa')




def DotMatrices(number_of_rows, number_of_cols, number_of_factors, user_factor_matrix, factor_item_matrix):
    predicted_ratings = np.zeros((number_of_rows, number_of_cols))
    sum = 0
    for i in range(0, number_of_rows):
        for j in range(0, number_of_cols):
            for k in range(0, number_of_factors):
                sum = sum + (user_factor_matrix[i][k] * factor_item_matrix[k][j])
            predicted_ratings[i][j] = sum
            sum = 0
    return predicted_ratings




def PopulateUserReviewMatrix(user_list, review_list, item_list):
    number_of_users = len(user_list)
    number_of_items = len(item_list)
    user_review_matrix = np.zeros((number_of_users + 1, number_of_items + 1))
    for review in review_list:
        user_review_matrix[review.userId][review.itemId] = review.rating
    return user_review_matrix
        
            

def readReviews(filename):
    f = open("ml-100k/" + filename, "r")
    
    a = f.readlines()
    length = len(a)
    review_list = []
    for x in range(0, length):
        fields = a[x].split()
        if int(fields[0]) <= 100 and int(fields[1]) <= 100:
            review_list.append(Review(fields[0], fields[1], fields[2]))
    
    return review_list

def readUsers():
    f = open('ml-100k/u.user', 'r')
    
    a = f.readlines()
    length = len(a)
    user_list = []
    for x in range(0, 100):
        fields = a[x].split('|')
        user_list.append(User(fields[0] ,fields[1], fields[2], fields[3], fields[4]))

    return user_list

def readItems():
    f = open('ml-100k/u.item', 'r')
    
    a = f.readlines()
    length = len(a)
    item_list = []
    for x in range(0, 100):
        fields = a[x].split('|')
        item_list.append(Item(fields[0], fields[1]))
    
    return item_list
if __name__ == "__main__":
    main()
