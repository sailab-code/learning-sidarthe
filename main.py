from data_manager import DataManager
from git_manager import GitManager

if __name__ == '__main__':
    git_manager = GitManager()
    data_manager = DataManager()
    print(data_manager.get_all_national())