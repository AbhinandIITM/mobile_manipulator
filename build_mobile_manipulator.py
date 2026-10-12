import xml.etree.ElementTree as ET
import mujoco

def build_mobile_manipulator():
    # 1. Parse ur5e_gripper.xml and j100_asm.xml
    ur_tree = ET.parse('ur5e/ur5e_gripper.xml')
    ur_root = ur_tree.getroot()

    j100_tree = ET.parse('j100/j100_asm/j100_asm.xml')
    j100_root = j100_tree.getroot()

    # 2. Rename UR5e base body to 'ur5e_base' to prevent collision with j100 'base'
    ur_wb = ur_root.find('worldbody')
    ur5e_base = None
    for b in ur_wb.findall('body'):
        if b.attrib.get('name') == 'base':
            ur5e_base = b
            ur5e_base.attrib['name'] = 'ur5e_base'
            break

    if ur5e_base is None:
        raise RuntimeError("Could not find body 'base' in ur5e_gripper.xml")

    # Mount UR5e on Jackal top deck facing forward over front bumper (-X)
    ur5e_base.attrib['pos'] = "0.0 0.0 0.20"
    ur5e_base.attrib['euler'] = "0.0 0.0 -1.570796"
    if 'quat' in ur5e_base.attrib:
        del ur5e_base.attrib['quat']

    # Add contact excludes between chassis/top and arm to eliminate internal repulsion
    contact = ur_root.find('contact')
    if contact is None:
        contact = ET.SubElement(ur_root, 'contact')
    for b_chassis in ['chassis', 'top']:
        for b_arm in ['ur5e_base', 'shoulder_link', 'upper_arm_link']:
            contact.append(ET.Element('exclude', attrib={'body1': b_chassis, 'body2': b_arm}))

    # 3. Rename excludes / equality / contacts referencing 'base' in UR5e if any
    for elem in ur_root.iter('exclude'):
        if elem.attrib.get('body1') == 'base':
            elem.attrib['body1'] = 'ur5e_base'
        if elem.attrib.get('body2') == 'base':
            elem.attrib['body2'] = 'ur5e_base'

    for elem in ur_root.iter('connect'):
        if elem.attrib.get('body1') == 'base':
            elem.attrib['body1'] = 'ur5e_base'
        if elem.attrib.get('body2') == 'base':
            elem.attrib['body2'] = 'ur5e_base'

    # 4. Get J100 base body
    j_wb = j100_root.find('worldbody')
    j100_base = None
    for b in j_wb.findall('body'):
        if b.attrib.get('name') == 'base':
            j100_base = b
            j100_base.attrib['name'] = 'chassis'
            break

    if j100_base is None:
        raise RuntimeError("Could not find body 'base' in j100_asm.xml")

    # 5. Merge Assets
    ur_asset = ur_root.find('asset')
    j_asset = j100_root.find('asset')
    for elem in j_asset:
        # If it's a mesh, update relative file path from ur5e/ to j100/
        if elem.tag == 'mesh':
            if elem.attrib.get('name') == 'base':
                elem.attrib['name'] = 'j100_base_mesh'
            if 'file' in elem.attrib:
                elem.attrib['file'] = f"../../j100/j100_asm/{elem.attrib['file']}"
        # Avoid duplicate asset names if any
        existing = ur_asset.find(f".//*[@name='{elem.attrib.get('name', '')}']")
        if existing is None and elem.attrib.get('name') != 'groundplane':
            ur_asset.append(elem)

    # Update J100 geom referencing 'base' mesh (before attaching arm)
    for geom in j100_base.iter('geom'):
        if geom.attrib.get('mesh') == 'base':
            geom.attrib['mesh'] = 'j100_base_mesh'

    # Attach UR5e base to Jackal chassis
    ur_wb.remove(ur5e_base)
    j100_base.append(ur5e_base)

    # 6. Put Jackal chassis into worldbody
    ur_wb.append(j100_base)

    # 7. Merge Actuators
    ur_act = ur_root.find('actuator')
    j_act = j100_root.find('actuator')
    if j_act is not None:
        # Prepend Jackal wheel actuators so wheel controls are indices 0..3
        for elem in reversed(list(j_act)):
            ur_act.insert(0, elem)

    # 8. Merge Sensors
    ur_sensor = ur_root.find('sensor')
    if ur_sensor is None:
        ur_sensor = ET.SubElement(ur_root, 'sensor')
    j_sensor = j100_root.find('sensor')
    if j_sensor is not None:
        for elem in j_sensor:
            ur_sensor.append(elem)

    # Set root model name
    ur_root.attrib['model'] = 'mobile_manipulator'

    # Save to ur5e/mobile_manipulator.xml
    output_path = 'ur5e/mobile_manipulator.xml'
    ET.indent(ur_tree, space="  ")
    ur_tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Successfully generated {output_path}")

    # Validate with MuJoCo
    m = mujoco.MjModel.from_xml_path(output_path)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    print(f"Validation SUCCESS! nq: {m.nq}, nv: {m.nv}, nu: {m.nu}")
    for i in range(m.nu):
        act_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  Actuator {i}: {act_name}")

if __name__ == '__main__':
    build_mobile_manipulator()
