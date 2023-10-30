# GDS Public

Knowledge mining project for Lifelike. "Forked" from SBRG/GDS including only the NW-arangodb branch.

If you have just started working on the project, see the below sections for guidance on getting things set up.

**Conda Users**: Follow the instructions [here](#setup-for-conda-users) first!

## Setup pipenv

First, install [pipenv](https://pypi.org/project/pipenv/) on your machine if you haven't already. [These](https://pipenv.pypa.io/en/latest/install/) instructions will walk you through installation.

Once you have successfully installed pipenv, run the following in the root directory of the project:

```bash
pipenv sync
```

This will create a new python virtual environment, and install all packages specified in the pipfile in that environment.

**Note:** You may need to add pipenv to your system's `PATH` variable. [This](https://pipenv.pypa.io/en/latest/install/#pragmatic-installation-of-pipenv) section of the installation steps above address modifying your `PATH`.

## Configure Editor to use Python Virtual Environment

You may want to configure your IDE to automatically hook into the virtual environment you created earlier. See the different sections below for guidance on how to set this up for your specific editor.

### VS Code

First, create a folder in the root of the project called `.vscode`, if you haven't already. In that folder, create a file called `launch.json`. In the file, add the following:

```json
{
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        }
    ]
}
```

If the `configurations` list property already exists, just add the additional object to it.

Next, in the bottom left of your editor, you should see something like:

![Screen Shot 2021-10-07 at 3 25 28 PM](https://user-images.githubusercontent.com/12260867/136470566-2650e9a6-a031-4135-a5ca-c0eaf68973fb.png)

Click on the "Python" section, and a window should open at the top of the editor. Select the `GDS` option, and then select the location of your desired virtual environment from the list, or enter it manually.

Now when you open a VS Code terminal your virtual environment will automatically be activated.

Next, open up your workspace settings file and add the following to the `settings` property:

```json
    "python.pythonPath": "/path/to/your/virtualenvs/your-virtual-env/bin/python",
```

Your workspace should now recognize all packages installed in the virtualenv you specified.

**Note:** If you aren't sure where your virtualenv is located, you should be able to find it by running: `pipenv --venv`. This will output the absolute path to the currently active virtualenv.

### PyCharm

Follow the instructions [here](https://www.jetbrains.com/help/pycharm/pipenv.html#pipenv-existing-project) to setup a pipenv environment.

## Setup for Conda Users

1. Create an environment using the Anaconda Prompt: `conda create -n <environment-name> python=3.9`
2. In Anaconda Navigator switch to that environment at the dropdown menu after “Applications on”
3. Install for that environment Spyder (You can also install PyCharms if you prefer).
4. Open Spyder for the gdsaddn environment.
5. On the IPython console run: `pip install /path/to/GDS-repo`. This finds and installs all dependencies automatically.

Also, see [this](https://github.com/SBRG/GDS/blob/master/docs/GDS_Conda_Install_Notes.docx) document, which the above instructions are sourced from.

## Setup Editor to Auto-Format Python on Save

This project uses [black](https://black.readthedocs.io/en/stable/index.html) to format code via the command line. However, you can also setup your editor to use black formatting automatically upon saving a file. If you want to setup your editor to do so, please follow the instructions [here](https://black.readthedocs.io/en/stable/integrations/editors.html).

VS Code users may also find [these](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0) instructions helpful. Make sure you save the new settings to your workspace, and _not_ to your user settings!

## Configure Pre-Commit

Before your commits can automatically be checked for style/formatting issues, you need to setup your git hooks to use `pre-commit`. Make sure you've installed pipenv and setup your virtual environment as described above, and then simply run `pre-commit install`. This will update your `.git/hooks/pre-commit` script to run `pre-commit` any time you commit changes.
