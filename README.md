# Collaborative Sequential Recommendations by Considering Personalized Time Pattern and Contrastive Learning

This repository contains the implementation of a senior project titled **Collaborative Sequential Recommendations by Considering Personalized Time Pattern and Contrastive Learning**. The project studies next-item recommendation by combining **sequential modeling**, **cross-user collaborative reasoning**, and **personalized temporal weighting** in a unified framework.

## Overview

Sequential recommendation models are widely used to predict the next item a user is likely to interact with by learning from the chronological order of past interactions. However, many existing methods mainly focus on a single user’s sequence and do not explicitly model collaborative information across users. Traditional collaborative recommendation methods, on the other hand, capture shared behavioral patterns across users but often treat interactions as static and ignore interaction order. This project addresses both limitations by introducing a framework that jointly models **order-aware collaboration** and **personalized temporal dynamics**.

The proposed framework is motivated by two main issues in prior work:

1. **Relative-order misalignment across users**: users may interact with similar items, but at different positions in their chronological sequences.
2. **Temporal-gap misalignment**: the same item transition can occur under very different time intervals for different users, and those differences should not be treated as equivalent. 

## Objectives

The project is designed around two main objectives:

- **Order-aware Collaborative Reasoning**  
  To align and compare user interaction sequences at corresponding positions, preserving chronological order instead of relying only on shared items. 

- **Personalized Time Weighting**  
  To model individualized temporal sensitivity using a monotonic time-decay function learned via isotonic regression, so the importance of interactions can adapt to each user’s own activity rhythm. 

## Proposed Method

The framework consists of three main modules: 

### 1. Temporal Sequence Encoding
A Transformer-based encoder is used to convert each user’s chronological interaction history into contextualized sequence representations. A learnable `[CLS]` token is prepended so that the model can derive a sequence-level user representation while still preserving token-level contextual embeddings for each interaction position. 

### 2. Personalized Time Pattern and Consecutive Item Correlation
This module models personalized temporal behavior by learning a **user-specific monotonic time-decay function** through isotonic regression. It then computes **consecutive item correlation** within each user sequence and compares these aligned pairwise patterns across users using cosine similarity. This enables collaborative similarity to be computed in an order-aware and time-aware manner. 

### 3. Contrastive Sequential Learning and Prediction
The final module uses contrastive learning to bring together users with similar sequential-temporal behavior while separating dissimilar users. This contrastive objective is combined with a ranking-based prediction objective for next-item recommendation. At inference time, candidate items are ranked and the top-K items are returned as recommendations. 

## Dataset and Evaluation

The experiments in this project are conducted on **MovieLens-1M**, a benchmark dataset for recommender systems with **6,040 users** and **3,706 items**. The recommendation task is formulated as **next-item prediction** based on each user’s ordered interaction history. 
The main baseline used in the evaluation is **GRU4Rec**, which represents a conventional recurrent sequential recommender that models next-item prediction mainly from a single user’s sequence. 

Evaluation is based on ranking metrics, especially:

- **NDCG@K**
- **HR@K** 

According to the reported results in the project document, both variants of the proposed framework outperform GRU4Rec, and the version **with time decay** achieves the best overall NDCG performance. 

## Why This Project Matters

This project aims to bridge the gap between **sequential recommendation** and **collaborative filtering**. Instead of modeling user behavior only within isolated sequences, the framework reasons over aligned behavior across users while also respecting each user’s own temporal rhythm. As a result, the method is intended to produce recommendations that are:

- **order-aware**
- **time-aware**
- **collaboratively informed** 

This makes the approach relevant to practical recommendation scenarios such as **e-commerce**, **media streaming**, and other dynamic environments where user behavior evolves over time. 
## How to Run

Install the required dependencies first:

```bash
pip install -r requirements.txt
