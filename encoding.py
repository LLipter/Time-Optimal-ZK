# encoding.py
# (C) 2021 Sasha Golovnev
# linear-time encoding with constant relative distance

import math
import random
import numpy as np
from datetime import datetime

# Auxiliary class to store sparse matrices as lists of non-zero elements
# (I couldn't figure out a simple way to use scipy's sparse matrices for our goals.)
class SparseMatrix:

    # A matrix of size n x m,
    # a[i] is a list of pairs (j, val) corresponding to matrix with A[i, j] = val
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.a = []
        for i in range(n):
            self.a.append([])

    # Add a non-zero element to the matrix
    def add(self, i, j, val):
        if val == 0:
            return
        assert 0 <= i < self.n
        assert 0 <= j < self.m
        self.a[i].append((j, val))


    # Multiply a vector x of length self.n by this matrix
    def multiply(self, x):
        assert len(x) == self.n
        y = [0] * self.m
        for i in range(self.n):
            for (j, val) in self.a[i]:
                y[j] += x[i] * val
        return [val % p for val in y]

    # Generate a matrix of size n x m, where each row has d *distinct* random positions
    # filled with random non-zero elements of the field
    @staticmethod
    def generate_random(n, m, d, p):
        matrix = SparseMatrix(n, m)
        for i in range(n):
            # print(m, d)
            if m < d:
                indices = [i for i in range(m)]
            else:
                indices = random.sample(range(m), d)
            for j in indices:
                matrix.add(i, j, random.randint(1, p-1))
        return matrix

    def transpose(self):
        a_prime = []
        for i in range(self.m):
            a_prime.append([])
        for i in range(self.n):
            for j, value in self.a[i]:
                a_prime[j].append((i, value))
        self.n, self.m = self.m, self.n
        self.a = a_prime

    def print(self):
        print(self.n, self.m)
        for i in range(self.n):
            row = [0] * self.m
            for j, value in self.a[i]:
                row[j] = value
            print(row)


def field_list_add(A, B):
    assert len(A) == len(B)
    return [field_add(a, b) for a, b in zip(A, B)]

# field size, prime p ~ 2^{256}
# p = 90589243044481683682024195529420329427646084220979961316176257384569345097147
# p = 69345097147
p = 97
n = 2**7

# the following parameters are taken from Table 1 of the write-up.
alpha = 0.238
beta = 0.1205
r = 1.72

# delta is the distance of the code, not really needed for the encoding procedure, used for testing.
delta = beta / r

# multiply two field elements
def field_mult(a, b):
    return (a % p) * (b % p) % p


# sum up two field elements
def field_add(a, b):
    return (a + b) % p


# the binary entropy function
def H(x):
    assert 0 < x < 1
    return -x*math.log(x,2)-(1-x)*math.log(1-x,2)


# I don't implement an efficient encoding by Reed-Solomon, because I assume we already have it.
# Instead I just generate a Vandermonde matrix and multiply it by the input vector x.
# This procedure takes a vector x of length l = len(x), and outputs a vector of length m.
def reed_solomon(x, m):
    l = len(x)
    # Reed-Solomon requires the field size to be at least the length of the output
    assert p > m
    y = [0] * m
    for i in range(1, m+1):
        a = 1
        for j in range(l):
            y[i-1] = field_add(y[i-1], field_mult(a, x[j]))
            a = field_mult(a, i)
    return y

def vandermonde(k, n):
    matrix = SparseMatrix(k, n)
    ai = 1
    for i in range(n):
        value = 1
        for j in range(k):
            matrix.add(j, i, value)
            value = field_mult(value, i+1)
    return matrix



# The code is given by two lists of sparse matrices, precodes and postcodes.
# Let n0 = n, n1 = alpha n0, n2 = alpha n1,...., nt = alpha n_{t-1},
# where nt is the first value <= 20.
#
# Each list of matrices will have t matrices.
# The i-th matrix in precodes has dimensions ni x (alpha * ni), where alpha is the parameter fixed above.
# The i-th matrix in postcodes has dimensions (alpha * r * ni) x ((r - 1 - alpha * r) * n_i),
# where alpha and r are the parameters fixed above.
#
# Each matrix in *precodes* is just a random sparse matrix sampled as follows:
# In each row of the matrix we pick cn distinct random indices.
# Then each of the sampled positions of the matrix gets a random non-zero element of the field.
# Since all the matrices are sparse, we store them as lists of non-zero elements.
#
# Matrices in *postcode* are sampled in the same way as in precodes with dn non-zeros per row instead of cn.
def generate(n):
    precodes = []
    postcodes = []
    i = 0
    ni = n
    while ni > 20:
        # the current precode matrix has dimensions ni x mi
        mi = math.ceil(ni * alpha)
        # the current postcode matrix has dimensions niprime x miprime
        niprime = math.ceil(r * mi)
        miprime = math.ceil(r * ni) - ni - niprime
        # print(1.2 * beta / alpha)
        # the sparsity of the precode matrix is cn
        cn = math.ceil(min(
            max(1.2 * beta * ni, beta * ni +3),
            (110/ni + H(beta) + alpha * H(1.2 * beta / alpha)) / (beta * math.log(alpha / (1.2 * beta), 2))
        ))
        precode = SparseMatrix.generate_random(ni, mi, cn, p)
        precodes.append(precode)

        # the sparsity of the postcode matrix is dn
        mu = r - 1 - r * alpha
        nu = beta + alpha * beta + 0.03
        # print(nu)
        # print(mu)
        dn = math.ceil(min(
            ni * (2 * beta + (r - 1 + 110/ni)/math.log(p, 2) ),
            (r * alpha * H(beta / r) + mu * H(nu / mu) + 110/ni) / (alpha * beta * math.log(mu / nu, 2))
        ))
        postcode = SparseMatrix.generate_random(niprime, miprime, dn, p)
        postcodes.append(postcode)

        i += 1
        ni = math.ceil(ni * alpha)
    return precodes, postcodes


# The encoding procedure.
# If the length of x is at most 20, then we just encode the vector by Reed-Solomon.
# Otherwise we take the next pair of matrices from precodes and postcodes
# y = x * precode
# z = recursive encoding of y
# v = z * postcode
# the resulting encoding is (x, z, v)
def encode(x, code, shift = 0):
    if len(x) <= 20:
        return reed_solomon(x, math.ceil(r * len(x)))
    precodes, postcodes = code
    assert precodes[shift].n == len(x)
    y = precodes[shift].multiply(x)
    z = encode(y, code, shift+1)
    assert postcodes[shift].n == len(z)
    v = postcodes[shift].multiply(z)
    # here '+' denotes the list concatenation operator.
    return x + z + v

def reverse_encode(w, code, shift = 0):
    # print("len(w): ", len(w))
    if len(w) <= 31.4:
        v_matrix = vandermonde(math.floor(len(w)/r), len(w))
        # v_matrix.print()
        # print("math.ceil(len(w)/r)", math.ceil(len(w)/r), "len(w)", len(w))
        v_matrix.transpose()
        # v_matrix.print()
        return v_matrix.multiply(w)
    precodes, postcodes = code

    x_len = precodes[shift].n
    v_len = postcodes[shift].m
    z_len = len(w) - x_len - v_len

    # print("x_len:", x_len)
    # print("z_len:", z_len)
    # print("v_len:", v_len)

    x = w[:x_len]
    w = w[x_len:]
    z = w[:z_len]
    w = w[z_len:]
    v = w[:v_len]
    w = w[v_len:]

    postcodes[shift].transpose()
    z_prime = postcodes[shift].multiply(v)
    postcodes[shift].transpose()
    z = field_list_add(z, z_prime)

    y = reverse_encode(z, code, shift+1)

    precodes[shift].transpose()
    x_prime = precodes[shift].multiply(y)
    precodes[shift].transpose()
    x = field_list_add(x, x_prime)
    
    return x
    

def test_reverse():
    code = generate(n)
    linear_matrix = SparseMatrix(n, math.ceil(n * r))
    for i in range(n):
        x = [0 for j in range(n)]
        x[i] = 1
        # print(x)
        encoded = encode(x, code)
        # print(encoded)
        # print(len(encoded))
        for j, val in enumerate(encoded):
            linear_matrix.add(i, j, val)
        print(i)
    linear_matrix.transpose()

    data = []
    for _ in range(math.ceil(n * r)):
        data.append(random.randint(0, p-1))
    # data = [i for i in range(math.ceil(n * r))]
    
    rev_enc_1 = reverse_encode(data, code)
    rev_enc_2 = linear_matrix.multiply(data)

    for a, b in zip(rev_enc_1, rev_enc_2):
        if a != b:
            print("test failed")
            return 
    print("test passed")

# # example
# code = generate(n)

# for _ in range(10):
#     x = []
#     # generate a random vector x of length n without using range(p):
#     for _ in range(n):
#         x.append(random.randint(0, p-1))
#     encoded = encode(x, code)
#     hammingWeight = sum(1 for element in encoded if element != 0)
#     print(hammingWeight/len(encoded) >= delta)


class RBRGraph:
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.graph = []
        for i in range(n):
            self.graph.append([])

    def add(self, i, j):
        self.graph[i].append(j)

    def print(self):
        print(self.n, self.d)
        for i in range(self.n):
            print(self.graph[i])

    def check(self):
        assert(len(self.graph) == self.n)
        for l in self.graph:
            assert(len(l) == self.d)

    def redistribute(self, x):
        assert(self.n == len(x))
        result = []
        for i in range(self.n):
            for j in range(self.d):
                idx = self.graph[i][j]
                result.append(x[idx])
        return result
    
    @staticmethod
    def generate_random(n, d):
        graph = RBRGraph(n, d)
        # regular
        permute = [i for i in range(n)]
        for i in range(d):
            random.shuffle(permute)
            for j in range(n):
                graph.add(j, permute[j])
        # # left regular
        # for i in range(n):
        #     for j in range(d):
        #         graph.add(i, random.randint(0, n - 1))
        return graph


def step1():
    n_prime = math.ceil(n * r)
    graph = RBRGraph.generate_random(n_prime, d)
    # graph.check()

    code = generate(n)
    x = [i for i in range(n)]
    encoded = encode(x, code)
    print("code_len:", len(encoded))

    redistributed = graph.redistribute(encoded)
    print(len(redistributed))
    return redistributed

def step2(redistributed):
    redistributed_len = len(redistributed)
    assert(len(redistributed) / d == len(redistributed) // d)
    no_block = len(redistributed) // d
    matrix = SparseMatrix(redistributed_len, redistributed_len)
    print(no_block)
    for i in range(no_block):
        # print(i)
        for j in range(d):
            for k in range(d):
                val = random.randint(1, p-1)
                idx_x = i * d + j
                idx_y = i * d + k
                matrix.add(idx_x, idx_y, val)
                if idx_x >= redistributed_len or idx_y >= redistributed_len:
                    print("!")
    print("matrix constructed")
    randomlized = matrix.multiply(redistributed)
    print("randomlized_len:", len(randomlized))
    return randomlized

def step3(randomlized):
    # r = 1.57
    n_prime = math.ceil(len(randomlized) / r)
    print("n_prime:", n_prime)
    while len(randomlized) < math.ceil(n_prime * r):
        randomlized.append(0)
    assert(len(randomlized) == math.ceil(n_prime * r))
    code = generate(n_prime)
    result = reverse_encode(randomlized, code)
    print(len(result))
    return result

def mul_inverse(x):
    return pow(int(x), -1, p)


# write rows in row echelon form
def upper_triangular(M):
    # move all zeros to buttom of matrix
    M = np.concatenate((M[np.any(M != 0, axis=1)], M[np.all(M == 0, axis=1)]), axis=0)

    # iterate over matrix rows
    for i in range(0, M.shape[0]):

        # initialize row-swap iterator
        j = 1

        # select pivot value
        pivot = M[i][i]

        # find next non-zero leading coefficient
        while pivot == 0 and i + j < M.shape[0]:
            # perform row swap operation
            M[[i, i + j]] = M[[i + j, i]]

            # incrememnt row-swap iterator
            j += 1

            # get new pivot
            pivot = M[i][i]

        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            break

        # extract row
        row = M[i]

        # get 1 along the diagonal
        M[i] = (row * mul_inverse(pivot)) % p

        # iterate over remaining rows
        for j in range(i + 1, M.shape[0]):
            # subtract current row from remaining rows
            M[j] = M[j] - M[i] * M[j][i]
            M[j] %= p

    for i in range(1, M.shape[0]):
        if M[i, i] == 0:
            continue

        for j in range(0, i):
            ratio = mul_inverse(M[i, i]) * (M[j, i] % p) % p
            M[j] -= ratio * M[i]
            M[j] %= p

    # return upper triangular matrix
    return M

def rank(M):
    A = upper_triangular(M)
    print(A)
    cnt = 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if A[i, j] != 0:
                cnt += 1
                break
    print(cnt)
    return cnt

def ns_basis(M):
    # r_m = rank(M)
    # r_m = rank(M[:, :-1])

    msg_len = M.shape[0]
    code_len = M.shape[1]
    A = M.transpose()
    # print(A)
    E = np.identity(code_len, dtype=np.longlong)
    # print(E)
    X = np.concatenate((A, E), axis=1, dtype=np.longlong)
    # print(X)

    XX = upper_triangular(X)
    # print(XX)
    XX = XX[:, msg_len:]
    # print(XX)

    # for i in range(code_len):
    #     print(M.dot(XX[i,:]) % p)

    XXX = XX[msg_len:, :]
    # print(XXX)

    return XXX

if __name__ == '__main__':
    # redistributed = step1()
    # randomlized = step2(redistributed)
    # result = step3(randomlized)



    
    random.seed(datetime.now())
    # random.seed(0)

    code = generate(n)
    x = [i for i in range(n)]
    encoded_t = encode(x, code)
    code_len = len(encoded_t)

    # for _ in range(10):
    #     x = []
    #     # generate a random vector x of length n without using range(p):
    #     for _ in range(n):
    #         x.append(random.randint(0, p-1))
    #     encoded = encode(x, code)
    #     hammingWeight = sum(1 for x, y in zip(encoded, encoded_t) if x !=y)
    #     # print(hammingWeight)
    #     print(hammingWeight/len(encoded) >= delta)

    print(code_len)


    data = np.zeros([n, code_len], dtype=np.longlong)
    for i in range(n):
        x = [0 for _ in range(n)]
        x[i] = 1
        encoded = encode(x, code)
        for j, val in enumerate(encoded):
            data[i, j] = val % p
            # data[i, j+code_len] = val % p
    # data[0, -1] = 1

    print(data.shape)
    print(data)

    nb = ns_basis(data)
    print(nb.shape)
    print(nb)

    min_t = 99999999999
    for i in range(code_len - n):
        # print(nb[i])
        cnt = 0
        for j in range(code_len):
            if nb[i, j] != 0:
                cnt += 1
        # print(i, cnt)
        min_t = min(min_t, cnt)
    
    print(min_t)

    # size = (10, 15)
    # data = np.zeros(size, dtype=np.longlong)
    # for i in range(size[0]):
    #     data[i, i] = 1
    # for i in range(size[0]):
    #     for j in range(size[0], size[1]-1):
    #         data[i, j] = random.randint(0, p-1)
    # data[0, size[1]-1] = 1
    # print(data)

    # nb = ns_basis(data)
    # print(nb)