import pickle

results_file = 'grid.pickle'
with open(results_file, 'rb') as f:
    grid = pickle.load(f)

def summarize_grid(grid, val_loss_threshold):
    """
    Print out the more promising hyperparameters from a search, according to
    validation loss.
    """
    best_val_loss = 1e9
    best_params = None
    items = sorted(grid, key = lambda k: min(grid[k]['history']['val_loss']))
    count = 0
    for frozen_key in items:
        count += 1
        value = grid[frozen_key]
        key = dict(frozen_key)
        val_loss = value['history']['val_loss']
        min_val_loss = min(val_loss)
        nb_epochs = len(val_loss)
        if min_val_loss < val_loss_threshold:
            print(key, min_val_loss, nb_epochs)
            print('cp', value['model_json'], 'model.json')
            print('cp', value['model_weights_h5'], 'model.h5')
            print('python drive.py model.json')
            print()
        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_params = key
    print('BEST:', best_params, best_val_loss)
    print(count)
summarize_grid(grid, 0.07)
