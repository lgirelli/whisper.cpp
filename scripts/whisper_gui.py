"""Simple Tkinter-based GUI to run whisper-cli on a selected audio file.

This helper expects the whisper-cli binary to be available at
``build/bin/whisper-cli`` and the large-v3-turbo model to live in ``models``.
Launch with ``python3 scripts/whisper_gui.py`` from the repository root.
"""

from pathlib import Path
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


REPO_ROOT = Path(__file__).resolve().parents[1]
BINARY_PATH = REPO_ROOT / "build" / "bin" / "whisper-cli"
MODEL_PATH = REPO_ROOT / "models" / "ggml-large-v3-turbo.bin"


class WhisperGUI:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("whisper-cli GUI")

        self.file_path: Path | None = None
        self.process: subprocess.Popen[str] | None = None

        self._build_widgets()

    def _build_widgets(self) -> None:
        padding = {"padx": 10, "pady": 5}

        self.file_label = tk.Label(self.master, text="Nessun file selezionato")
        self.file_label.grid(row=0, column=0, sticky="w", **padding)

        browse_button = tk.Button(
            self.master, text="Scegli file", command=self.select_file
        )
        browse_button.grid(row=0, column=1, sticky="e", **padding)

        self.run_button = tk.Button(
            self.master, text="Esegui whisper-cli", command=self.run_command, state=tk.DISABLED
        )
        self.run_button.grid(row=1, column=0, columnspan=2, sticky="ew", **padding)

        self.output_box = scrolledtext.ScrolledText(self.master, width=80, height=20)
        self.output_box.grid(row=2, column=0, columnspan=2, sticky="nsew", **padding)

        self.master.grid_rowconfigure(2, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=0)

    def select_file(self) -> None:
        initial_dir = (REPO_ROOT / "samples").as_posix()
        file_path = filedialog.askopenfilename(initialdir=initial_dir)
        if file_path:
            self.file_path = Path(file_path)
            self.file_label.config(text=str(self.file_path))
            self.run_button.config(state=tk.NORMAL)

    def run_command(self) -> None:
        if not self.file_path:
            messagebox.showwarning("File mancante", "Seleziona prima un file audio")
            return

        if not BINARY_PATH.exists():
            messagebox.showerror(
                "whisper-cli mancante",
                f"Binary non trovato in {BINARY_PATH}. Assicurati di aver eseguito la build.",
            )
            return

        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Modello mancante",
                f"Modello non trovato in {MODEL_PATH}. Scarica ggml-large-v3-turbo.bin in models/",
            )
            return

        self.output_box.delete("1.0", tk.END)
        self.run_button.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._execute_command, daemon=True)
        thread.start()

    def _execute_command(self) -> None:
        command = [
            str(BINARY_PATH),
            "-m",
            str(MODEL_PATH),
            "-f",
            str(self.file_path),
        ]

        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        except FileNotFoundError as error:
            self._append_output(f"Errore: {error}\n")
            self._on_process_end()
            return

        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._append_output(line)

        self.process.wait()
        self._append_output(f"\nComando terminato con codice {self.process.returncode}\n")
        self._on_process_end()

    def _append_output(self, text: str) -> None:
        self.output_box.insert(tk.END, text)
        self.output_box.see(tk.END)

    def _on_process_end(self) -> None:
        self.run_button.config(state=tk.NORMAL)


def main() -> None:
    root = tk.Tk()
    app = WhisperGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
