# PowerShell script to install VS Code extensions
$extensions = @(
    "4ops.terraform",
    "amazonwebservices.aws-toolkit-vscode",
    "dannysteenman.aws-terraform-extension-pack",
    "dannysteenman.iam-actions-snippets",
    "dannysteenman.iam-service-principal-snippets",
    "erkanp.excalidraw-new-file",
    "github.copilot",
    "github.copilot-chat",
    "hashicorp.terraform",
    "k--kato.intellij-idea-keybindings",
    "ms-python.debugpy",
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.vscode-python-envs",
    "ms-toolsai.jupyter",
    "ms-toolsai.jupyter-keymap",
    "ms-toolsai.jupyter-renderers",
    "ms-toolsai.vscode-jupyter-cell-tags",
    "ms-toolsai.vscode-jupyter-slideshow",
    "ogranny.md-template",
    "pomdtr.excalidraw-editor",
    "redhat.java",
    "tomoki1207.pdf",
    "visualstudioexptteam.intellicode-api-usage-examples",
    "visualstudioexptteam.vscodeintellicode",
    "vscjava.migrate-java-to-azure",
    "vscjava.vscode-gradle",
    "vscjava.vscode-java-debug",
    "vscjava.vscode-java-dependency",
    "vscjava.vscode-java-pack",
    "vscjava.vscode-java-test",
    "vscjava.vscode-java-upgrade",
    "vscjava.vscode-maven"
)

Write-Host "Installing VS Code extensions..."
foreach ($extension in $extensions) {
    Write-Host "Installing $extension..."
    code --install-extension $extension
}

Write-Host "Installation complete! Please restart VS Code to activate all extensions."