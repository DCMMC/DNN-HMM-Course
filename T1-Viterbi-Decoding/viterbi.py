#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Xingchen Song @ 2020-10-12

import numpy as np


class HMM:
    """Hidden Markov Model

    HMM with 3 states and 2 observation categories.

    Attributes:
        ob_category (list, with length 2): observation categories
        total_states (int): number of states, default=3
        pi (array, with shape (3,)): initial state probability
        A (array, with shape (3, 3)): transition probability. A.sum(axis=1) must be all ones.
                                      A[i, j] means transition prob from state i to state j.
                                      A.T[i, j] means transition prob from state j to state i.
        B (array, with shape (3, 2)): emitting probability, B.sum(axis=1) must be all ones.
                                      B[i, k] means emitting prob from state i to observation k.

    """

    def __init__(self):
        self.ob_category = ['THU', 'PKU']  # 0: THU, 1: PKU
        self.total_states = 3
        self.pi = np.array([0.2, 0.4, 0.4])
        self.A = np.array([[0.1, 0.6, 0.3],
                           [0.3, 0.5, 0.2],
                           [0.7, 0.2, 0.1]])
        self.B = np.array([[0.5, 0.5],
                           [0.4, 0.6],
                           [0.7, 0.3]])

    def forward(self, ob):
        """HMM Forward Algorithm.

        Args:
            ob (array, with shape(T,)): (o1, o2, ..., oT), observations

        Returns:
            fwd (array, with shape(T, 3)): fwd[t, s] means full-path forward probability torwards state s at
                                           timestep t given the observation ob[0:t+1].
                                           给定观察ob[0:t+1]情况下t时刻到达状态s的所有可能路径的概率和
            prob: the probability of HMM model generating observations.

        """
        T = ob.shape[0]
        fwd = np.zeros((T, self.total_states))

        # Begin Assignment

        # PUT YOUR CODE HERE.
        # Forward algorithm:
        # Input: O = o_1, o_2, \cdots, o_T, \lambda = {\pi, A, B}
        # > O: observations, \pi: initial probability, A: state transition matrix, B: emission matrix.
        # Output: P(O | \lambda)
        #   # Law of total prob for all valid hidden state seq Q = q_1, q_2, \cdots, q_T
        #   = \sum_{Q} P(O, Q | \lambda)
        #   = \sum_Q P(O | Q, \lambda) P(Q | \lambda) # Bayes
        #   = \sum_Q \prod_{t=1}^T B(q_t, o_t) \pi(q_1) \prod_{t=2}^T A(q_{t-1}, q_t)
        #
        # In matrix operation form:
        # \pi^{(0)} = \pi, Fwd(o_1) = \pi^{(1)} = \pi^{(0)} * B(:, o_1), P(o_1) = sum(Fwd(o_1))
        # ...
        # Fwd(o_1, \cdots, o_T) = \pi^{(T)} = \pi^{(T-1)} * B(:, o_T)
        # Output: P(o_1, \cdots, o_T) = sum(Fwd(o_1. \cdots, o_T))
        # Complexity: O(T * N^2)
        pi_t = self.pi
        for t, o_t in enumerate(ob):
            fwd[t] = pi_t * self.B[:, o_t]
            pi_t = fwd[t] @ self.A

        # End Assignment

        prob = fwd[-1, :].sum()

        return fwd, prob

    def backward(self, ob):
        """HMM Backward Algorithm.

        Args:
            ob (array, with shape(T,)): (o1, o2, ..., oT), observations

        Returns:
            bwd (array, with shape(T, 3)): bwd[t, s] means full-path backward probability torwards state s at
                                           timestep t given the observation ob[t+1::]
                                           给定观察ob[t+1::]情况下t时刻到达状态s的所有可能路径的概率和
            prob: the probability of HMM model generating observations.

        """
        T = ob.shape[0]
        bwd = np.zeros((T, self.total_states))

        # Begin Assignment

        # PUT YOUR CODE HERE.
        # The formula is similar to forward algorithm, but inducted backward.
        # Complexity: O(T * N^2)
        beta_t = np.ones(self.total_states)
        for t, o_t in enumerate(reversed(ob)):
            bwd[-1-t] = beta_t.T
            beta_t = self.A @ (self.B[:, o_t] * beta_t)

        # End Assignment

        prob = (bwd[0, :] * self.B[:, ob[0]] * self.pi).sum()

        return bwd, prob

    def viterbi(self, ob):
        """Viterbi Decoding Algorithm.

        Args:
            ob (array, with shape(T,)): (o1, o2, ..., oT), observations

        Variables:
            delta (array, with shape(T, 3)): delta[t, s] means max probability torwards state s at
                                             timestep t given the observation ob[0:t+1]
                                             给定观察ob[0:t+1]情况下t时刻到达状态s的概率最大的路径的概率
            phi (array, with shape(T, 3)): phi[t, s] means prior state s' for delta[t, s]
                                           给定观察ob[0:t+1]情况下t时刻到达状态s的概率最大的路径的t-1时刻的状态s'

        Returns:
            best_prob: the probability of the best state sequence
            best_path: the best state sequence

        """
        T = ob.shape[0]
        delta = np.zeros((T, self.total_states))
        phi = np.zeros((T, self.total_states), np.int)
        best_prob, best_path = 0.0, np.zeros(T, dtype=np.int)

        # Begin Assignment

        # PUT YOUR CODE HERE.

        # End Assignment

        best_path[T-1] = delta[T-1, :].argmax(0)
        best_prob = delta[T-1, best_path[T-1]]
        for t in reversed(range(T-1)):
            best_path[t] = phi[t+1, best_path[t+1]]

        return best_prob, best_path


if __name__ == "__main__":
    model = HMM()
    observations = np.array([0, 1, 0, 1, 1])  # [THU, PKU, THU, PKU, PKU]
    fwd, p = model.forward(observations)
    print(p, '\n', fwd)
    bwd, p = model.backward(observations)
    print(p, '\n', bwd)
    prob, path = model.viterbi(observations)
    print(prob, '\n', path)
