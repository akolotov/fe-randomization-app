# Development with VSCode Devcontainers and GitHub Codespaces

## GitHub SSH Key

Before re-opening in the container, you may find it useful to configure SSH authorization. To do this:

1. Ensure you have SSH access to GitHub configured on your local machine.

2. Open `.devcontainer/devcontainer.json`.

3. In the mounts section adjust the paths if your SSH keys are stored in a different location.

    ```json
    "mounts": [
    "source=${localEnv:HOME}/.ssh/config,target=/home/vscode/.ssh/config,type=bind,consistency=cached",
    "source=${localEnv:HOME}/.ssh/id_rsa,target=/home/vscode/.ssh/id_rsa,type=bind,consistency=cached"
    ],
    ```

4. Use `git update-index --assume-unchanged .devcontainer/devcontainer.json` to prevent the changes to `devcontainer.json` from appearing in git status and VS Code's Source Control. To undo the changes, use `git update-index --no-assume-unchanged .devcontainer/devcontainer.json`.

## Additional Setup for Cursor.ai Users

If you're using Cursor.ai instead of VSCode, you may need to perform some additional setup steps. Please note that these changes will not persist after reloading the devcontainer, so you may need to repeat these steps each time you start a new session.

### Git Configuration

You may encounter issues when trying to perform Git operations from the terminal or the "Source Control" tab. To resolve this, set up your Git configuration inside the devcontainer:

1. Open a terminal in your devcontainer.

2. Set your Git username:

    ```bash
    git config --global user.name "Your Name"
    ```

3. Set your Git email:

    ```bash
    git config --global user.email "your.email@example.com"
    ```

Replace "Your Name" and "`your.email@example.com`" with your actual name and email associated with your GitHub account.
