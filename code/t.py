import xml.etree.ElementTree as ET
import pandas as pd

tmx_file = r"D:/HUET/NAM 4/NAM 4 KI I/computer_vision/clean_repo/code/file.tmx"
output_file = "tmx_output.xlsx"

tree = ET.parse(tmx_file)
root = tree.getroot()

# Đăng ký namespace xml
ns = {"xml": "http://www.w3.org/XML/1998/namespace"}

data = []
for tu in root.findall(".//tu"):
    en_seg = tu.find(".//tuv[@xml:lang='en-US']/seg", namespaces=ns)
    vi_seg = tu.find(".//tuv[@xml:lang='vi-VN']/seg", namespaces=ns)
    en = en_seg.text if en_seg is not None else ""
    vi = vi_seg.text if vi_seg is not None else ""
    data.append([en, vi])

df = pd.DataFrame(data, columns=["English", "Vietnamese"])
df.to_excel(output_file, index=False)

print(f"✅ Xuất thành công: {output_file}")
