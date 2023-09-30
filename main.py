"""

Solving linear equations using the Gauss-Seidel method

Author: Dmytro Zuiev
Version: 1.0

"""

from math import sqrt
from random import randrange
import numpy



def get_lower_digits(number: int) -> str:
    """ The method recursively converts a number to its string equivalent 
    with subscript digits
    """    
    digits = ('₀','₁','₂','₃','₄','₅','₆','₇','₈','₉')
    if number // 10 == 0:
        return digits[number % 10]
    return get_lower_digits(number // 10) + digits[number % 10]


def print_message(index, message):
    """ The method displays a message to the console with the index 
    at the beginning of the line
    """
    lower_index = get_lower_digits(index)
    print('₍{0}₎ {1}'.format(lower_index, message))


def read_equation(index: int) -> tuple:
    """ The method reads the coefficients of a linear equation from the console
    
    Args:
        index -- serial number of the equation. 
        Output at the beginning of the line
    Returns:
        coefficients, result -- list of coefficients 
        and constant term of the equation
    """
    is_complete = False
    while not is_complete:
        try:
            coefficients, result = input('₍{}₎ '.format(get_lower_digits(index))).split('=', 1)
            coefficients = [float(a) for a in coefficients.split()]
            result = float(result)
            is_complete = True
        except (ValueError):
            print_message(index, 'Invalid format: use only rational numbers, single symbol \'=\', and a constant term after it')
    return coefficients, result


def read_equations_system() -> tuple:
    """ The method reads a system of linear equations from the console
    
    Returns:
        coefficients_matrix, results_vector -- matrix of coefficients 
        and vector of constant terms of the system of equations
    """
    coefficients_matrix = []
    results_vector = []
    rows_number = 1
    row_index = 1

    while row_index <= rows_number:
        coefficients, result = read_equation(row_index)
        
        if len(coefficients) == 0:
            print_message(row_index, 'Invalid format: enter coefficients')
            continue
        if row_index == 1:
            rows_number = len(coefficients)
        elif len(coefficients) != rows_number:
            print_message(row_index, 'The number of coefficients must match. Use 0 for missing parameters')
            continue
        
        coefficients_matrix.append(coefficients)
        results_vector.append(result)
        row_index += 1
        
    return coefficients_matrix, results_vector


def solve(coefficients: list, results: list, eps: float) -> list:
    """ The method solves a system of linear equations using the Gauss-Seidel method
    
    Args:
        coefficients -- coefficient matrix
        results -- constant terms vector
        eps -- approximation accuracy
    Returns:
        a vector of system roots
    Raises:
        OverflowError -- if the method does not converge
        ZeroDivisionError -- if there are zeros in the main diagonal that cannot be solved by rearranging rows
    """
    n = len(coefficients)
    x = None
    counter = 0
    COUNTER_LIMIT = n ** 2
    while x is None:
        try:
            x = [results[i] / coefficients[i][i] for i in range(n)]
        except ZeroDivisionError:
            """trying to solve the problem of diagonal zeros"""
            for i in range(n):
                if coefficients[i][i] != 0:
                    continue
                """shuffle the lines"""
                random_row = randrange(n)
                random_row = (random_row + 1) % n if random_row == i else random_row
                temp_row = coefficients[random_row]
                coefficients[random_row] = coefficients[i]
                coefficients[i] = temp_row
                temp_result = results[random_row]
                results[random_row] = results[i]
                results[i] = temp_result
        finally:
            counter += 1
            if counter > COUNTER_LIMIT:
                raise ZeroDivisionError('Could not resolve conflict with diagonal zeros')

    converge = False
    deviation = 0
    deviation_growth_count = 0
    DEVIATION_GROWTH_LIMIT = n
    while not converge:
        x_new = numpy.copy(x)
        for i in range(n):
            sum_new = sum(coefficients[i][j] * x_new[j] for j in range(i))
            sum_prev = sum(coefficients[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (results[i] - sum_new - sum_prev) / coefficients[i][i]
        new_deviation = sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n)))
        x = x_new

        if (new_deviation - deviation > -eps):
            deviation_growth_count += 1
        else:
            deviation_growth_count = 0
            
        deviation = new_deviation
        converge = deviation < eps
        
        """if the method does not converge"""
        if deviation_growth_count > DEVIATION_GROWTH_LIMIT:
            raise OverflowError('the Gauss-Seidel method does not converge')

    return x


def print_rules():
    """ The method outputs a description of the incoming data
    """    
    print('Enter the coefficients of the system equation separated by a space,')
    print('add the = symbol, write the constant term, and press Enter to enter the equation.')
    print('For example, a line: 1.2 -1 3 5 = 10')
    print('will be interpreted like 1.2х₁ – х₂ + 3х₃ + 5х₄ = 10')
    

def print_result(x_vector: list):
    """ The method outputs the resulting list of roots to the console
    """    
    print('Solution:')
    index = 1
    for x in x_vector:
        print('x{} = {}'.format(get_lower_digits(index), round(x, 2)))
        index += 1
        



if __name__ == '__main__':
    
    print_rules()
    while True:
        print('Enter system equations:')
        coefficients_matrix, results_vector = read_equations_system()
        try:
            x_vector = solve(coefficients_matrix, results_vector, .01)
            print_result(x_vector)
        except (OverflowError, ZeroDivisionError) as error:
            print(error)
            print('Attempt 2. What about other methods?')
            try:
                x_vector = numpy.linalg.solve(coefficients_matrix, results_vector)
                print_result(x_vector)
            except numpy.linalg.LinAlgError:
                print('Well. We were never able to find a solution. But at least we tried.')
