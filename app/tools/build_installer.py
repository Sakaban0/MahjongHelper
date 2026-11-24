"""Package the public ZIP as a per-user Windows Installer setup (Python 3.12)."""
import hashlib
import msilib as m
from msilib import schema, sequence
from pathlib import Path
import os
import subprocess
import tempfile
import uuid
import zipfile

from mahjong_helper import __version__
from .build_release import ROOT, ZIP_NAME, main as build_zip

FAMILY = uuid.UUID("6a2264b4-546c-4f76-a321-cc33707847ed")


def guid(name):
    return "{" + str(uuid.uuid5(FAMILY, name)).upper() + "}"


def main():
    build_zip()
    out = ROOT / "dist"
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        source = work / "source"
        with zipfile.ZipFile(out / ZIP_NAME) as archive:
            archive.extractall(source)
        files = sorted(p for p in source.rglob("*") if p.is_file())
        digest = hashlib.sha256()
        for file in files:
            digest.update(file.relative_to(source).as_posix().encode())
            digest.update(hashlib.sha256(file.read_bytes()).digest())
        msi = work / "MahjongHelper.msi"
        db = m.init_database(str(msi), schema, "Mahjong Helper", guid(digest.hexdigest()),
                             __version__, "Mahjong Helper")
        summary = db.GetSummaryInformation(1)
        summary.SetProperty(m.PID_WORDCOUNT, 10)  # compressed, per-user privileges
        summary.Persist()
        del summary
        m.add_tables(db, sequence)
        m.add_data(db, "Property", [("UpgradeCode",guid("family")),("SecureCustomProperties","OLDHELPER")])
        m.add_data(db, "Upgrade", [(guid("family"),None,__version__,None,512,None,"OLDHELPER")])
        cab = m.CAB("payload")
        feature = m.Feature(db,"Helper","Mahjong Helper",None,1)
        base = m.Directory(db,cab,None,str(source),"TARGETDIR","SourceDir")
        local = m.Directory(db,cab,base,".","LocalAppDataFolder",".")
        dest = m.Directory(db,cab,local,".","APPDIR","MahjongHelper")
        dirs = {source:dest}
        for i,folder in enumerate(sorted(p for p in source.rglob("*") if p.is_dir())):
            parent = dirs[folder.parent]
            dirs[folder] = m.Directory(db,cab,parent,folder.name,"D"+str(i),parent.make_short(folder.name)+"|"+folder.name)
        for i,file in enumerate(files):
            directory = dirs[file.parent]
            component = "C"+str(i)
            # Preserve user settings and registered images during upgrade and uninstall.
            attributes = 0x90 if file.is_relative_to(source / "json") else 0
            directory.start_component(component,feature,attributes,file.name,guid(file.relative_to(source).as_posix()))
            key = directory.add_file(file.name,src=str(file))
            if file.name == "Start-Desktop.vbs":
                m.add_data(db,"Directory",[("ProgramMenuFolder","TARGETDIR",".")])
                m.add_data(db,"Shortcut",[("Launch","ProgramMenuFolder","MJHELPER|Mahjong Helper",component,
                                            "[SystemFolder]wscript.exe",'"[#'+key+']"',None,None,None,None,1,"APPDIR")])
        for name in ("json","debug"):
            directory = dirs[source/name]
            directory.start_component("Folder"+name,feature,0,uuid=guid("folder/"+name))
            m.add_data(db,"CreateFolder",[(directory.logical,directory.component)])
        cab.commit(db)
        db.Close()
        setup = work / "MahjongHelper-Setup.exe"
        compiler = Path(os.environ["WINDIR"])/"Microsoft.NET/Framework64/v4.0.30319/csc.exe"
        subprocess.run([str(compiler),"/nologo","/target:winexe","/reference:System.Windows.Forms.dll",
                        "/resource:"+str(msi)+",payload","/out:"+str(setup),
                        str(ROOT/"tools/SetupLauncher.cs")],check=True)
        setup.replace(out/setup.name)
        print(out/setup.name)


if __name__ == "__main__":
    main()
