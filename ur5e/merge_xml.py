import xml.etree.ElementTree as ET

ur5e_tree = ET.parse('ur5e.xml')
ur5e_root = ur5e_tree.getroot()

gripper_tree = ET.parse('assets/robotiq_2f85/2f85.xml')
gripper_root = gripper_tree.getroot()

d435i_tree = ET.parse('assets/realsense_d435i/d435i.xml')
d435i_root = d435i_tree.getroot()

def rename_classes_and_materials(root, prefix):
    # Rename defaults
    for elem in root.iter('default'):
        cls_name = elem.attrib.get('class')
        if cls_name in ['visual', 'collision']:
            elem.attrib['class'] = f"{prefix}_{cls_name}"
            
    # Rename materials
    for elem in root.iter('material'):
        mat_name = elem.attrib.get('name')
        if mat_name:
            elem.attrib['name'] = f"{prefix}_{mat_name}"

    # Rename references in geoms
    for elem in root.iter('geom'):
        cls_name = elem.attrib.get('class')
        if cls_name in ['visual', 'collision']:
            elem.attrib['class'] = f"{prefix}_{cls_name}"
            
        mat_name = elem.attrib.get('material')
        if mat_name:
            elem.attrib['material'] = f"{prefix}_{mat_name}"

rename_classes_and_materials(gripper_root, "2f85")
rename_classes_and_materials(d435i_root, "d435i")

# Rename conflicting body 'base' in gripper
for elem in gripper_root.iter('body'):
    if elem.attrib.get('name') == 'base':
        elem.attrib['name'] = '2f85_base'

for elem in gripper_root.iter('exclude'):
    if elem.attrib.get('body1') == 'base':
        elem.attrib['body1'] = '2f85_base'
    if elem.attrib.get('body2') == 'base':
        elem.attrib['body2'] = '2f85_base'

# 1. Merge assets and rewrite file paths
ur5e_asset = ur5e_root.find('asset')

for elem in gripper_root.find('asset'):
    if 'file' in elem.attrib:
        elem.attrib['file'] = f"robotiq_2f85/assets/{elem.attrib['file']}"
    ur5e_asset.append(elem)

for elem in d435i_root.find('asset'):
    if 'file' in elem.attrib:
        elem.attrib['file'] = f"realsense_d435i/assets/{elem.attrib['file']}"
    ur5e_asset.append(elem)

# 2. Merge defaults
ur5e_default = ur5e_root.find('default')
for d in gripper_root.iter('default'):
    if d.attrib.get('class') == '2f85':
        ur5e_default.append(d)
        break

for d in d435i_root.iter('default'):
    if d.attrib.get('class') == 'd435i':
        ur5e_default.append(d)
        break

# 3. Add contact, tendon, equality, actuator to top level
for tag in ['contact', 'tendon', 'equality', 'actuator']:
    ur5e_tag = ur5e_root.find(tag)
    if ur5e_tag is None:
        ur5e_tag = ET.SubElement(ur5e_root, tag)
    
    gripper_tag = gripper_root.find(tag)
    if gripper_tag is not None:
        for elem in gripper_tag:
            ur5e_tag.append(elem)

# 4. Attach gripper body to wrist_3_link
def find_body(root, name):
    for body in root.iter('body'):
        if body.attrib.get('name') == name:
            return body
    return None

wrist_3 = find_body(ur5e_root, 'wrist_3_link')
gripper_base_mount = find_body(gripper_root, 'base_mount')

gripper_base_mount.attrib['pos'] = "0 0.1 0"
gripper_base_mount.attrib['quat'] = "-1 1 0 0"
wrist_3.append(gripper_base_mount)

# 5. Attach RealSense to gripper base
gripper_base = find_body(gripper_base_mount, '2f85_base')
d435i_body = find_body(d435i_root, 'd435i')

# The user screenshot showed it on the side when Y=0.04.
# Let's put it on the front (X=0.04).
# And rotate by 90 degrees around Z (1.5708) to make it horizontal, 
# then tilt down towards pinch by rotating around Y or X.
# We will use quat instead for explicit rotations if euler is confusing, but let's try euler.
# Let's rotate 90 deg around Z: euler="0 0 1.5708"
# Let's tilt down (towards Z) around Y: euler="0 -0.15 1.5708"
d435i_body.attrib['pos'] = "0.04 0 0.0"
d435i_body.attrib['euler'] = "0 -0.15 1.5708"

# Add a camera tag inside the d435i body
cam = ET.Element('camera', attrib={
    'name': 'd435i_color',
    'pos': '0 0.0175 0', # offset slightly based on new rotation
    'euler': '3.1415 0 0', # point it forward
    'fovy': '60'
})
d435i_body.append(cam)

gripper_base.append(d435i_body)

# Write out merged xml
ET.indent(ur5e_tree, space="  ")
ur5e_tree.write('ur5e_gripper.xml', encoding='utf-8', xml_declaration=True)
print("Merged XML saved to ur5e_gripper.xml")
