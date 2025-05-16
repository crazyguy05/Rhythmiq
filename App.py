import torch
from audiocraft.models import MusicGen
import torchaudio
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import shutil
import subprocess


class MusicGenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Music Generator")
        self.root.geometry("600x400")
        self.model = None
        self.playback_process = None  # Track ffplay process

        self.audio_dir = os.path.join(os.getcwd(), "musicgen_output")
        os.makedirs(self.audio_dir, exist_ok=True)
        self.audio_path = os.path.join(self.audio_dir, "output.wav")

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="🎵 Rythmiq: AI Music Generator", font=('Helvetica', 16, 'bold')).pack(pady=10)

        ttk.Label(main_frame, text="Tell me the rhythm of your day:").pack(anchor=tk.W)
        self.description_entry = tk.Text(main_frame, height=5, width=60, wrap=tk.WORD)
        self.description_entry.pack(pady=5)
        self.description_entry.insert("1.0", "80s synthwave with heavy bass, 120 BPM")

        ttk.Label(main_frame, text="Duration (seconds):").pack(anchor=tk.W)
        self.duration_slider = ttk.Scale(main_frame, from_=5, to=30, value=10, orient=tk.HORIZONTAL)
        self.duration_slider.pack(fill=tk.X, pady=5)
        self.duration_label = ttk.Label(main_frame, text="10")
        self.duration_label.pack()
        self.duration_slider.bind("<Motion>", self.update_duration_label)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)

        self.generate_btn = ttk.Button(button_frame, text="Generate Music", command=self.start_generation_thread)
        self.generate_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(button_frame, text="Play", command=self.play_audio, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(button_frame, text="Save", command=self.save_audio, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, pady=10)

    def update_duration_label(self, event):
        self.duration_label.config(text=f"{int(self.duration_slider.get())}")

    def load_model(self):
        if self.model is None:
            self.status_var.set("Loading model... (this may take 2-3 minutes)")
            self.root.update()
            self.model = MusicGen.get_pretrained('facebook/musicgen-small')
            self.status_var.set("Model ready")

    def generate_music(self):
        try:
            # Wait for previous playback to end
            if self.playback_process and self.playback_process.poll() is None:
                self.status_var.set("Waiting for playback to finish...")
                self.playback_process.wait()

            self.generate_btn.config(state=tk.DISABLED)
            description = self.description_entry.get("1.0", tk.END).strip()
            duration = int(self.duration_slider.get())

            if not description:
                messagebox.showerror("Error", "Please enter a description")
                return

            self.load_model()
            self.status_var.set("Generating music...")
            self.root.update()

            temp_path = os.path.join(self.audio_dir, "temp.wav")

            self.model.set_generation_params(
                use_sampling=True,
                top_k=250,
                duration=duration
            )

            audio_tensor = self.model.generate([description], progress=True)[0]
            torchaudio.save(temp_path, audio_tensor.detach().cpu(), 32000)

            if os.path.exists(self.audio_path):
                os.remove(self.audio_path)
            os.rename(temp_path, self.audio_path)

            self.status_var.set("Ready to play!")
            self.play_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.generate_btn.config(state=tk.NORMAL)

    def play_audio(self):
        try:
            if not os.path.exists(self.audio_path):
                messagebox.showerror("Error", "Please generate music first!")
                return

            # If there's already a playback process, terminate it
            if self.playback_process and self.playback_process.poll() is None:
                self.playback_process.terminate()

            self.status_var.set("Playing...")
            self.root.update()

            # Launch ffplay as subprocess and track it
            self.playback_process = subprocess.Popen([
                "ffplay", "-nodisp", "-autoexit", self.audio_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        except Exception as e:
            messagebox.showerror("Play Error", f"Failed to play: {str(e)}")

    def save_audio(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile="generated_music.wav"
        )
        if file_path:
            try:
                shutil.copy2(self.audio_path, file_path)
                self.status_var.set(f"Saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {str(e)}")

    def start_generation_thread(self):
        thread = threading.Thread(target=self.generate_music)
        thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGenApp(root)
    root.mainloop()
