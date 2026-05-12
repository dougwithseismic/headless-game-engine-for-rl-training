interface CameraToggleProps {
  /** Label shown on the button (e.g. current camera mode name). */
  mode: string;
  /** Called when the user clicks to switch camera modes. */
  onToggle: () => void;
}

/**
 * HTML overlay button for switching between camera modes.
 * Rendered outside the Canvas, positioned with CSS class `.camera-toggle`.
 */
export function CameraToggle({ mode, onToggle }: CameraToggleProps) {
  return (
    <div className="camera-toggle">
      <button className="ctrl-btn active" onClick={onToggle}>
        {mode}
      </button>
    </div>
  );
}
