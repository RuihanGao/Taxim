"""Extract every STEP leaf with assembly transforms; optional isolated CAD environment."""
from pathlib import Path
import argparse,json,re,hashlib
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Choose an empty CAD export directory')
    try:
        import cadquery as cq
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TDocStd import TDocStd_Document
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
        from OCP.TDF import TDF_Label,TDF_LabelSequence
        from OCP.TDataStd import TDataStd_Name
        from OCP.TopLoc import TopLoc_Location
        from OCP.IFSelect import IFSelect_RetDone
    except ImportError as e:raise SystemExit('STEP import needs requirements-cad.txt in an optional CAD environment; use run_pipeline.py --cad-python /path/to/that/python. '+str(e))
    def label_name(label):
        att=TDataStd_Name();return att.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(),att) else 'unnamed'
    a.output.mkdir(parents=True,exist_ok=True);doc=TDocStd_Document(TCollection_ExtendedString('Mini CAD'))
    reader=STEPCAFControl_Reader();reader.SetNameMode(True)
    if reader.ReadFile(str(a.input))!=IFSelect_RetDone or not reader.Transfer(doc):raise ValueError('Unable to read STEP assembly')
    st=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main());roots=TDF_LabelSequence();st.GetFreeShapes(roots);parts=[]
    def walk(label,location,parents):
        ref=TDF_Label()
        if st.GetReferredShape_s(label,ref):return walk(ref,location.Multiplied(st.GetLocation_s(label)),parents+[label_name(label)])
        children=TDF_LabelSequence()
        if st.GetComponents_s(label,children):
            for i in range(1,children.Length()+1):walk(children.Value(i),location,parents+[label_name(label)])
            return
        q=cq.Shape.cast(st.GetShape_s(label).Moved(location));name=label_name(label);idx=len(parts);bb=q.BoundingBox()
        vertices,faces=q.tessellate(.15,.2);v=np.asarray([p.toTuple() for p in vertices]);f=np.asarray(faces,dtype=np.int64)
        filename=f'{idx:03d}_'+re.sub('[^a-zA-Z0-9_-]','_',name)+'.npz';path=a.output/filename;np.savez_compressed(path,vertices=v,faces=f)
        rec=dict(index=idx,name=name,hierarchy=parents,mesh_file=filename,mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bounds_mm=[[bb.xmin,bb.ymin,bb.zmin],[bb.xmax,bb.ymax,bb.zmax]],faces=[])
        for face in q.Faces():
            kind=face.geomType()
            if kind not in ['PLANE','CYLINDER']:continue
            item=dict(kind=kind,center_mm=face.Center().toTuple(),area_mm2=face.Area())
            if kind=='PLANE':item['normal']=face.normalAt().toTuple()
            else:
                cyl=face._geomAdaptor().Cylinder();item.update(radius_mm=cyl.Radius(),axis_origin_mm=cyl.Location().Coord(),axis_direction=cyl.Axis().Direction().Coord())
            rec['faces'].append(item)
        parts.append(rec);print(idx,name,flush=True)
    for i in range(1,roots.Length()+1):walk(roots.Value(i),TopLoc_Location(),[])
    report=dict(source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),units='mm converted by OpenCascade STEP reader',all_leaf_instances_included=True,assembly_transforms_applied=True,parts=parts)
    (a.output/'assembly.json').write_text(json.dumps(report,indent=2)+'\n');print(a.output/'assembly.json')
if __name__=='__main__':main()
