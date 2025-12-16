import base64
import io
import json
import uuid

from IPython.display import HTML, display


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
    base_duration=83,    # ~12 FPS
    scale=1.0,           # Resolution scalar
    cell_width=None,     # <--- NEW: Explicit Width
    cell_height=None     # <--- NEW: Explicit Height
):
    
    # Generate unique IDs
    unique_id = uuid.uuid4().hex
    container_id = f"container_{unique_id}"
    
    # --- 1. Determine Dimensions ---
    # If dimensions aren't provided, try to detect from the first context image, 
    # otherwise fallback to 256.
    ref_w, ref_h = 256, 256
    
    # Try to grab reference from context
    if context_rgbs and len(context_rgbs) > 0:
        ref_w, ref_h = context_rgbs[0].width, context_rgbs[0].height
    # Try to grab reference from samples if context is empty
    elif sample_rgbs and len(sample_rgbs) > 0 and len(sample_rgbs[0]) > 0:
        ref_w, ref_h = sample_rgbs[0][0].width, sample_rgbs[0][0].height

    # Override with user arguments if provided
    final_w = cell_width if cell_width is not None else ref_w
    final_h = cell_height if cell_height is not None else ref_h
    
    # Apply the 'scale' factor to the base resolution (what actually gets sent over network)
    # and the initial display size
    base_px_w = int(final_w * scale)
    base_px_h = int(final_h * scale)

    # --- 2. Data Preparation ---
    animation_data = {}
    
    def process_frames(frames, is_mask=False):
        """Converts PIL frames to base64 with explicit resizing."""
        if not frames: return []
        b64_list = []
        fmt = 'PNG' if is_mask else 'JPEG'
        
        for img in frames:
            # Resize to the calculated base dimensions
            # This ensures all cells match the requested cell_width/height exactly
            if img.width != base_px_w or img.height != base_px_h:
                img = img.resize((base_px_w, base_px_h))
            
            buffer = io.BytesIO()
            if fmt == 'JPEG':
                img.convert('RGB').save(buffer, format='JPEG', quality=85)
            else:
                img.save(buffer, format='PNG')
            
            b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            b64_list.append(f"data:image/{fmt.lower()};base64,{b64_str}")
        return b64_list

    # --- 3. HTML Structure & CSS ---
    
    # We use CSS variables for BOTH width and height now
    css_w_var = f"--grid-w-{unique_id}"
    css_h_var = f"--grid-h-{unique_id}"
    
    # Inline references to these vars
    style_dim = f"width: var({css_w_var}); height: var({css_h_var});"
    style_w_only = f"width: var({css_w_var}); min-width: var({css_w_var});"

    def get_anim_cell_html(frames, is_mask=False):
        if not frames: return ""
        cell_id = f"anim_{unique_id}_{uuid.uuid4().hex}"
        processed = process_frames(frames, is_mask)
        animation_data[cell_id] = processed
        first_frame = processed[0] if processed else ""
        
        return f"""
            <img id="{cell_id}" src="{first_frame}" 
                 style="{style_dim} object-fit: contain; display:block; margin:0; padding:0;" />
        """

    bg_color = "#333333"       
    text_color = "#ffffff"     
    accent_color = "#4db8ff"
    border_color = "#444444"
    separator_style = f"border-right: 1px solid {border_color};" 
    
    fn_speed = f"setSpeed_{unique_id}"
    fn_scale = f"setScale_{unique_id}"

    rows_config = [
        ("RGB",     context_rgbs,    sample_rgbs,    False),
        ("Seg",     context_segs,    sample_segs,    True),
        ("Depth",   context_depths,  sample_depths,  False),
        ("Normals", context_normals, sample_normals, False)
    ]

    def make_btn(fn_name, group_name, value, label, is_default=False):
        weight = "bold" if is_default else "normal"
        return f"""<button class="btn-{group_name}-{unique_id}" 
                           onclick="{fn_name}({value}, this)" 
                           style="cursor:pointer; padding: 2px 8px; font-weight:{weight}; 
                                  background: #444; color: #fff; border: 1px solid #555; margin-right: 2px;">
                           {label}
                   </button>"""

    # Initialize HTML container with the CSS variables defined inline
    html = f"""
    <div id="{container_id}" 
         style="font-family: sans-serif; background-color: {bg_color}; padding: 10px; 
                border: 1px solid {border_color}; width: fit-content; 
                {css_w_var}: {base_px_w}px; 
                {css_h_var}: {base_px_h}px;">
        
        <div style="margin-bottom: 10px; display: flex; flex-wrap: wrap; align-items: center; gap: 15px; color: {text_color}; font-size: 13px;">
            <div style="display:flex; align-items:center; gap:5px;">
                <span style="font-weight:bold; color: {accent_color};">SPEED:</span>
                {make_btn(fn_speed, "speed", 4.0, "0.25x")}
                {make_btn(fn_speed, "speed", 1.0, "1x", True)}
                {make_btn(fn_speed, "speed", 0.5, "2x")}
            </div>
            <div style="display:flex; align-items:center; gap:5px; padding-left: 10px; border-left: 1px solid #555;">
                <span style="font-weight:bold; color: {accent_color};">SCALE:</span>
                {make_btn(fn_scale, "scale", 0.5, "0.5x")}
                {make_btn(fn_scale, "scale", 1.0, "1x", True)}
                {make_btn(fn_scale, "scale", 1.5, "1.5x")}
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
                               {separator_style} {style_w_only} text-align: center;">
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
    
    for seed in seeds:
        html += f"""
            <th style="padding: 5px 0; color: {text_color}; font-size: 12px; font-weight: normal;
                       background-color: {bg_color}; border: 1px solid {border_color}; 
                       {style_w_only} text-align: center;">
                seed={seed}
            </th>
        """
    html += "</tr></thead><tbody>"

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
                       {separator_style} {style_w_only}">
                {get_anim_cell_html(ctx_frames, is_mask)}
            </td>
        """
        for i, _ in enumerate(seeds):
            frames = sample_list[i] if i < len(sample_list) else []
            html += f"""
                <td style="padding: 0; margin: 0; line-height: 0; border: 1px solid {border_color}; 
                           {style_w_only}">
                    {get_anim_cell_html(frames, is_mask)}
                </td>
            """
        html += "</tr>"

    html += "</tbody></table></div></div>"

    # --- 4. JavaScript Engine ---
    json_data = json.dumps(animation_data)
    
    js_script = f"""
    <script>
    (function() {{
        const animData = {json_data};
        const baseDuration = {base_duration};
        const uniqueId = "{unique_id}";
        
        // Base dimensions passed from Python
        const baseW = {base_px_w};
        const baseH = {base_px_h};
        
        const containerId = "{container_id}";
        let timer = null;
        let currentFrameIdx = 0;
        
        function startLoop(multiplier) {{
            if (timer) clearInterval(timer);
            const interval = baseDuration * multiplier;
            const disp = document.getElementById('fps_display_' + uniqueId);
            if(disp) disp.innerText = (Math.round(1000 / interval * 10) / 10) + " FPS";

            timer = setInterval(() => {{
                currentFrameIdx++;
                for (const [id, frames] of Object.entries(animData)) {{
                    const imgEl = document.getElementById(id);
                    if (imgEl && frames.length > 0) {{
                        imgEl.src = frames[currentFrameIdx % frames.length];
                    }}
                }}
            }}, interval);
        }}

        function setActiveBtn(btnElement, className) {{
            if (btnElement) {{
                const allBtns = document.getElementsByClassName(className);
                for (let i = 0; i < allBtns.length; i++) {{
                    allBtns[i].style.fontWeight = "normal";
                    allBtns[i].style.background = "#444";
                }}
                btnElement.style.fontWeight = "bold";
                btnElement.style.background = "#666";
            }}
        }}

        window["{fn_speed}"] = function(multiplier, btnElement) {{
            startLoop(multiplier);
            setActiveBtn(btnElement, "btn-speed-" + uniqueId);
        }};

        // Scale controls now update both width and height variables
        window["{fn_scale}"] = function(scaleMult, btnElement) {{
            const newW = Math.floor(baseW * scaleMult);
            const newH = Math.floor(baseH * scaleMult);
            
            const container = document.getElementById(containerId);
            container.style.setProperty("--grid-w-" + uniqueId, newW + "px");
            container.style.setProperty("--grid-h-" + uniqueId, newH + "px");
            
            setActiveBtn(btnElement, "btn-scale-" + uniqueId);
        }};

        startLoop(1.0);
    }})();
    </script>
    """
    
    display(HTML(html + js_script))