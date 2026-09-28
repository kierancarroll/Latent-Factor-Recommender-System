

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def csv_to_ndarray(filename):
    with open(filename) as f:
        # Skip header
        f.readline()

        data_list = []
        for line in f:
            data_list.append([int(s) for s in line.split(',')])

    data_numpy = np.array(data_list)

    return data_numpy


def rec_load_tiny_data():
    """ Load tiny training dataset

        Returns tuple of length 3: (num_users, num_items, ratings_train_tiny)
        where num_users and num_items are both ints
        and ratings_train_tiny is an ndarray with one row per rating
        each row has three columns containing integers: [user_index, book_index, rating]
        Indices are zero-indexed.
    """
    ratings_train_tiny = csv_to_ndarray('data/ratings_train_tiny.csv')
    
    num_users = np.max(ratings_train_tiny[:, 0]) + 1
    num_items = np.max(ratings_train_tiny[:, 1]) + 1

    return (num_users, num_items, ratings_train_tiny)


def rec_load_data():
    """ Load training and validation datasets

        Returns tuple of length 4: (num_users, num_items, ratings, book_titles)
        where num_users and num_items are both ints
        and ratings is an ndarray with one row per rating
        each row has three columns containing integers: [user_index, book_index, rating]
        Indices are zero-indexed.
        book_titles is a list containing the book titles
    """
    ratings = csv_to_ndarray('data/ratings_train.csv')
    book_titles = list(pd.read_csv('data/books.csv')['title'])
    
    num_users = np.max(ratings[:, 0]) + 1
    num_items = np.max(ratings[:, 1]) + 1

    ratings[:, 2] = ratings[:, 2] - np.mean(ratings[:, 2])
    
    return (num_users, num_items, ratings, book_titles)

def visualize_embeddings(user_embeddings, book_embeddings,
                         user_indices_to_display, item_indices_to_display, book_titles=None):
    """ Plot the embeddings of the users with the embedding of the given book.

        user_indices: list of integers corresponding to the user indices in the same
            order of user_embeddings
        user_embeddings: (num_users, 2) numpy ndarray containing learned embeddings of users
        book_embeddings: (2,) numpy ndarray which is the embedding of the book
        book_title: string of the title of the book

        Returns: matplotlib figure plotting embeddings
    """
    min_alpha = 0.1
    default_alpha = 0.6

    fontsize = 14
    title = f'Embedded Users and Items'

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')

    # Add x and y axis lines
    # Z order in the back
    ax.axhline(0, color='black', lw=1, zorder=0)
    ax.axvline(0, color='black', lw=1, zorder=0)

    # Plotting embedded users
    for i in user_indices_to_display:
        emb = user_embeddings[i]
        point_color = 'tab:blue'
        point_size = 20
        line_color = 'tab:blue'
        line_width = 1
        label = f'User {i}'
        line_zorder = 1
        point_zorder = 2
        alpha = 1

        # Vector line
        ax.plot([0, emb[0]], [0, emb[1]], line_color, lw=line_width, alpha=alpha*0.5,
                zorder=line_zorder)

        # Vector point
        ax.scatter(emb[0], emb[1], s=point_size, c=point_color,
                zorder=point_zorder, alpha=alpha*0.5)

        # Offset text in the same direction as the vector
        num_chars = len(label)
        offset_scale = 0.9
        horiz_scale = 1.0 + 0.25*num_chars
        offset_shift = np.array([-num_chars*0.25-.05, -0.375])
        xy_offset = offset_scale * emb/np.linalg.norm(emb)
        xy_offset[0] *= horiz_scale
        xy_offset += offset_shift

        ax.annotate(label, (emb[0], emb[1]),
                    xytext=xy_offset, textcoords='offset fontsize',
                    fontsize=fontsize, color=point_color, alpha=alpha*0.7,
                    zorder=point_zorder)

    # Plotting the book embedding
    for j in item_indices_to_display:

        point_color = 'tab:purple'
        line_color = 'tab:purple'

        if book_titles is not None:
            book_title = book_titles[j]
            label = f'Book {j}: {book_title}'
        else:
            label = f'Book {j}'

        # Vector line
        ax.plot([0, book_embeddings[j, 0]], [0, book_embeddings[j, 1]], line_color, lw=line_width, alpha=alpha*0.7,
                zorder=line_zorder)

        # Vector point
        ax.scatter(book_embeddings[j, 0], book_embeddings[j, 1], s=point_size, c=point_color,
                zorder=point_zorder, alpha=alpha)

        # Offset text in the same direction as the vector
        num_chars = len(label)
        offset_scale = 0.9
        horiz_scale = 1.0 + 0.25*num_chars
        offset_shift = np.array([-num_chars*0.25-.05, -0.375])
        xy_offset = offset_scale * book_embeddings[j]/np.linalg.norm(book_embeddings[j])
        xy_offset[0] *= horiz_scale
        xy_offset += offset_shift

        ax.annotate(label, (book_embeddings[j, 0], book_embeddings[j, 1]),
                    xytext=xy_offset, textcoords='offset fontsize',
                    fontsize=fontsize, color=point_color, alpha=alpha,
                    zorder=point_zorder)

    max_vector_mag = np.max(np.linalg.norm(user_embeddings, axis=1))
    max_vector_mag = np.max((np.max(np.linalg.norm(book_embeddings, axis=1)), max_vector_mag))
    max_vector_mag *= 1.1
    ax.set_xlim(-max_vector_mag, max_vector_mag)
    ax.set_ylim(-max_vector_mag, max_vector_mag)

    ax.set_title(title)
