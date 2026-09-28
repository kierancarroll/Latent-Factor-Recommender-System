from utils import visualize_embeddings, rec_load_data
from latent_factor_model import latent_factor_loss, latent_factor_sgd

num_users, num_items, ratings, book_titles = rec_load_data()

K = 2
alpha = 0.01
num_epochs = 100

U, V = latent_factor_sgd(ratings, num_users, num_items, K, alpha, num_epochs, print_loss=True)

loss_train = latent_factor_loss(ratings, U, V)
print(f'Loss: {loss_train:.5f} (Expected loss: {0.18855})')

visualize_embeddings(U, V, list(range(20)),  list(range(0, 100, 20)), book_titles)