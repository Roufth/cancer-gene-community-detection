"""
Slim down ocd_model_artifacts.pkl untuk Streamlit Cloud.
Hapus 'model' (Word2Vec gensim object) karena tidak dipakai Streamlit
dan menghindari dependency gensim + Python version mismatch.

Cara jalankan:
    python slim_pickle.py

Atau dari Colab/notebook setelah load artifacts:
    Jalankan kode di __main__ section.
"""
import pickle
import os

PKL_PATH = 'ocd_model_artifacts.pkl'

if __name__ == '__main__':
    print(f'Loading: {PKL_PATH}')
    with open(PKL_PATH, 'rb') as f:
        artifacts = pickle.load(f)

    print(f'Original keys: {list(artifacts.keys())}')
    original_size = os.path.getsize(PKL_PATH) / (1024 * 1024)
    print(f'Original size: {original_size:.2f} MB')

    # Buang 'model' (gensim Word2Vec) — tidak dipakai Streamlit, hanya butuh embeddings array
    keys_to_remove = ['model']
    for key in keys_to_remove:
        if key in artifacts:
            del artifacts[key]
            print(f'  Removed: {key}')

    print(f'Slim keys: {list(artifacts.keys())}')

    # Backup original (kalau perlu rollback)
    backup_path = PKL_PATH + '.backup'
    if not os.path.exists(backup_path):
        os.rename(PKL_PATH, backup_path)
        print(f'Backup disimpan: {backup_path}')
    else:
        print(f'Backup sudah ada: {backup_path} (tidak ditimpa)')

    # Save slim version
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

    slim_size = os.path.getsize(PKL_PATH) / (1024 * 1024)
    print(f'Slim size    : {slim_size:.2f} MB (hemat {original_size - slim_size:.2f} MB)')
    print('\nSiap untuk Streamlit Cloud. Push lagi:')
    print('  git add ocd_model_artifacts.pkl')
    print('  git commit -m "Slim pickle: remove gensim Word2Vec model"')
    print('  git push')
