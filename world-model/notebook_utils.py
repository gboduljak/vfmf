import base64
import io
import json
import uuid

from IPython.display import HTML, display
from PIL import Image


def display_context_animations(dataset, max_frames=4, fps=6, scale=1.0):
    import base64
    import io
    import json
    import uuid

    from IPython.display import HTML, display

    unique_id = uuid.uuid4().hex
    container_id = f"anim_container_{unique_id}"

    # ------------------------------------------------------------------
    # Convert PIL frames → base64
    # ------------------------------------------------------------------
    animation_data = {}

    def encode_frames(frames):
        out = []
        for img in frames[:max_frames]:
            if scale != 1.0:
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)),
                )

            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            out.append(f"data:image/jpeg;base64,{b64}")
        return out

    for i, seq in enumerate(dataset):
        animation_data[f"seq_{unique_id}_{i}"] = {
            "frames": encode_frames(seq),
            "index": i,
        }

    json_data = json.dumps(animation_data)
    interval_ms = int(1000 / fps)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------
    html = f"""
    <div id="{container_id}" style="display:flex; gap:8px; align-items:flex-start;">
    """

    for key, val in animation_data.items():
        html += f"""
        <div style="
            display:flex;
            flex-direction:column;
            border:1px solid #333;
            font-family:monospace;
        ">
            <div style="
                height:48px;
                background:black;
                color:white;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:16px;
                flex-shrink:0;
            ">
                {val["index"]}
            </div>
            <img id="{key}"
                 src="{val["frames"][0]}"
                 style="display:block;" />
        </div>
        """

    html += "</div>"

    # ------------------------------------------------------------------
    # JS animation loop
    # ------------------------------------------------------------------
    js = f"""
    <script>
    (function() {{
        const data = {json_data};
        let frame = 0;

        setInterval(() => {{
            frame++;
            for (const [id, obj] of Object.entries(data)) {{
                const frames = obj.frames;
                if (!frames.length) continue;
                const img = document.getElementById(id);
                if (img) {{
                    img.src = frames[frame % frames.length];
                }}
            }}
        }}, {interval_ms});
    }})();
    </script>
    """

    display(HTML(html + js))


def display_controllable_grid(
    seeds, 
    sample_rgbs, sample_segs, sample_depths, sample_normals,
    context_rgbs, context_segs, context_depths, context_normals,
    base_duration=83,  # ~83ms per frame = 12 FPS
    scale=1.0          # Base resolution scaler (affects file size/quality)
):
    
    # Generate unique IDs for isolation
    unique_id = uuid.uuid4().hex
    container_id = f"container_{unique_id}"
    
    # --- 1. Data Preparation & JS Storage ---
    animation_data = {}
    
    def process_frames(frames, is_mask=False):
        """Converts PIL frames to a list of base64 strings with bilinear scaling."""
        if not frames: return []
        b64_list = []
        fmt = 'PNG' if is_mask else 'JPEG'
        
        for img in frames:
            # Server-side resizing (Fundamental Resolution)
            if scale != 1.0:
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), resample=Image.BILINEAR)
            
            buffer = io.BytesIO()
            if fmt == 'JPEG':
                img.convert('RGB').save(buffer, format='JPEG', quality=85)
            else:
                img.save(buffer, format='PNG')
            
            b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            b64_list.append(f"data:image/{fmt.lower()};base64,{b64_str}")
        return b64_list

    # --- 2. HTML Structure with CSS Variables ---
    
    # Calculate base pixel width (Resolution)
    ref_width = 128
    if context_rgbs and len(context_rgbs) > 0:
         ref_width = context_rgbs[0].width
    
    base_px = int(ref_width * scale)
    
    # We use a CSS variable for width so the JS can resize columns and images simultaneously
    css_width_var = f"var(--grid-width-{unique_id})"
    
    def get_anim_cell_html(frames, is_mask=False):
        if not frames: return ""
        
        cell_id = f"anim_{unique_id}_{uuid.uuid4().hex}"
        processed = process_frames(frames, is_mask)
        animation_data[cell_id] = processed
        first_frame = processed[0] if processed else ""
        
        # Image width is bound to the CSS variable
        return f"""
            <img id="{cell_id}" src="{first_frame}" 
                 style="width:{css_width_var}; display:block; margin:0; padding:0;" />
        """

    bg_color = "#333333"       
    text_color = "#ffffff"     
    accent_color = "#4db8ff"
    border_color = "#444444"
    separator_style = f"border-right: 1px solid {border_color};" 
    
    # Scoped JS function names
    fn_speed = f"setSpeed_{unique_id}"
    fn_scale = f"setScale_{unique_id}"

    rows_config = [
        ("RGB",     context_rgbs,    sample_rgbs,    False),
        ("Seg",     context_segs,    sample_segs,    True),
        ("Depth",   context_depths,  sample_depths,  False),
        ("Normals", context_normals, sample_normals, False)
    ]

    # Helper to generate buttons
    def make_btn(fn_name, group_name, value, label, is_default=False):
        weight = "bold" if is_default else "normal"
        # We pass 'this' so JS can handle the bolding logic
        return f"""<button class="btn-{group_name}-{unique_id}" 
                           onclick="{fn_name}({value}, this)" 
                           style="cursor:pointer; padding: 2px 8px; font-weight:{weight}; 
                                  background: #444; color: #fff; border: 1px solid #555; margin-right: 2px;">
                           {label}
                   </button>"""

    # Initialize HTML
    # Note: We define the default --grid-width value in the container's style
    html = f"""
    <div id="{container_id}" 
         style="font-family: sans-serif; background-color: {bg_color}; padding: 10px; 
                border: 1px solid {border_color}; width: fit-content; 
                --grid-width-{unique_id}: {base_px}px;">
        
        <div style="margin-bottom: 10px; display: flex; flex-wrap: wrap; align-items: center; gap: 15px; color: {text_color}; font-size: 13px;">
            
            <div style="display:flex; align-items:center; gap:5px;">
                <span style="font-weight:bold; color: {accent_color};">SPEED:</span>
                {make_btn(fn_speed, "speed", 4.0, "0.25x")}
                {make_btn(fn_speed, "speed", 2.0, "0.5x")}
                {make_btn(fn_speed, "speed", 1.0, "1x", True)}
                {make_btn(fn_speed, "speed", 0.5, "2x")}
            </div>

            <div style="display:flex; align-items:center; gap:5px; padding-left: 10px; border-left: 1px solid #555;">
                <span style="font-weight:bold; color: {accent_color};">SCALE:</span>
                {make_btn(fn_scale, "scale", 0.5, "0.5x")}
                {make_btn(fn_scale, "scale", 1.0, "1x", True)}
                {make_btn(fn_scale, "scale", 1.5, "1.5x")}
                {make_btn(fn_scale, "scale", 2.0, "2x")}
            </div>

            <span id="fps_display_{unique_id}" style="margin-left:auto; color: #888; font-size: 12px;">{int(1000/base_duration)} FPS</span>
        </div>

        <div style="overflow-x: auto;">
        <table style="border-collapse: collapse; border-spacing: 0; margin: 0; padding: 0; 
                      background-color: {bg_color}; border: 1px solid {border_color};">
            <thead>
                <tr>
                    <th style="background-color: {bg_color}; border: 1px solid {border_color};"></th>
                    <th style="padding: 10px 0; color: {accent_color}; font-weight: bold; font-size: 14px;
                               background-color: {bg_color}; border: 1px solid {border_color};
                               {separator_style} width: {css_width_var}; max-width: {css_width_var}; text-align: center;">
                        CONTEXT
                    </th>
                    <th colspan="{len(seeds)}" 
                        style="padding: 10px; color: {accent_color}; font-weight: bold; font-size: 14px;
                               background-color: {bg_color}; border: 1px solid {border_color}; text-align: center;">
                        SAMPLED FUTURES
                    </th>
                </tr>
                <tr>
                    <th style="background-color: {bg_color}; border: 1px solid {border_color};"></th>
                    <th style="background-color: {bg_color}; border: 1px solid {border_color}; {separator_style}"></th>
    """
    
    # Seed Headers: Width bound to CSS Var
    for seed in seeds:
        html += f"""
            <th style="padding: 5px 0; color: {text_color}; font-size: 12px; font-weight: normal;
                       background-color: {bg_color}; border: 1px solid {border_color}; 
                       width: {css_width_var}; min-width: {css_width_var}; text-align: center;">
                seed={seed}
            </th>
        """
    html += "</tr></thead><tbody>"

    # --- Populate Data Rows ---
    for label, ctx_frames, sample_list, is_mask in rows_config:
        html += "<tr>"
        html += f"""
            <td style="padding: 10px; font-weight: bold; text-align: center; color: {text_color}; 
                       background-color: {bg_color}; border: 1px solid {border_color}; vertical-align: middle;">
                {label}
            </td>
        """
        html += f"""
            <td style="padding: 0; margin: 0; line-height: 0; border: 1px solid {border_color}; 
                       {separator_style} width: {css_width_var};">
                {get_anim_cell_html(ctx_frames, is_mask)}
            </td>
        """
        for i, _ in enumerate(seeds):
            frames = sample_list[i] if i < len(sample_list) else []
            html += f"""
                <td style="padding: 0; margin: 0; line-height: 0; border: 1px solid {border_color}; 
                           width: {css_width_var};">
                    {get_anim_cell_html(frames, is_mask)}
                </td>
            """
        html += "</tr>"

    html += "</tbody></table></div></div>"

    # --- 3. JavaScript Player Engine ---
    json_data = json.dumps(animation_data)
    
    js_script = f"""
    <script>
    (function() {{
        const animData = {json_data};
        const baseDuration = {base_duration};
        const uniqueId = "{unique_id}";
        const basePx = {base_px};
        const containerId = "{container_id}";
        
        let timer = null;
        let currentFrameIdx = 0;
        
        // --- Animation Loop ---
        function startLoop(multiplier) {{
            if (timer) clearInterval(timer);
            const interval = baseDuration * multiplier;
            
            // Update FPS
            const fps = Math.round(1000 / interval * 10) / 10;
            const disp = document.getElementById('fps_display_' + uniqueId);
            if(disp) disp.innerText = fps + " FPS";

            timer = setInterval(() => {{
                currentFrameIdx++;
                for (const [id, frames] of Object.entries(animData)) {{
                    const imgEl = document.getElementById(id);
                    if (imgEl && frames.length > 0) {{
                        const frameSrc = frames[currentFrameIdx % frames.length];
                        imgEl.src = frameSrc;
                    }}
                }}
            }}, interval);
        }}

        // --- Helper: UI Logic for Buttons ---
        function setActiveBtn(btnElement, className) {{
            if (btnElement) {{
                const allBtns = document.getElementsByClassName(className);
                for (let i = 0; i < allBtns.length; i++) {{
                    allBtns[i].style.fontWeight = "normal";
                    allBtns[i].style.background = "#444";
                }}
                btnElement.style.fontWeight = "bold";
                btnElement.style.background = "#666"; // visual feedback
            }}
        }}

        // --- Public: Speed Control ---
        window["{fn_speed}"] = function(multiplier, btnElement) {{
            startLoop(multiplier);
            setActiveBtn(btnElement, "btn-speed-" + uniqueId);
        }};

        // --- Public: Scale Control ---
        window["{fn_scale}"] = function(scaleMult, btnElement) {{
            const newSize = Math.floor(basePx * scaleMult);
            const container = document.getElementById(containerId);
            // Update the CSS variable; the browser handles layout reflow instantly
            container.style.setProperty("--grid-width-" + uniqueId, newSize + "px");
            
            setActiveBtn(btnElement, "btn-scale-" + uniqueId);
        }};

        // Start at 1.0x speed
        startLoop(1.0);
    }})();
    </script>
    """
    
    display(HTML(html + js_script))