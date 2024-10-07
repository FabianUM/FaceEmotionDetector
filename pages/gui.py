import tkinter as tk

def create_gui(start_detection_callback, exit_program_callback):
    # Crear la ventana principal
    window = tk.Tk()
    window.title("Detector de Emociones")
    window.geometry("300x200")  # Ajustar tamaño de la ventana
    window.configure(bg="lightblue")  # Establecer el color de fondo

    # Centrar la ventana
    window.eval('tk::PlaceWindow . center')

    # Botón para iniciar el escaneo
    start_button = tk.Button(window, text="Iniciar Escaneo", command=start_detection_callback, bg="yellow")
    start_button.pack(pady=20)

    # Botón para salir del programa
    exit_button = tk.Button(window, text="Cerrar Programa", command=lambda: exit_program_callback(window), bg="yellow")
    exit_button.pack(pady=20)

    # Ejecutar la ventana principal
    window.mainloop()
