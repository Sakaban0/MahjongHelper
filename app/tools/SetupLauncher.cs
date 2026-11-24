using System;
using System.IO;
using System.Diagnostics;
using System.Reflection;
using System.Windows.Forms;

class SetupLauncher {
    [STAThread]
    static int Main() {
        if (MessageBox.Show("Mahjong Helper をこのユーザー用にインストールしますか？", "Mahjong Helper",
                MessageBoxButtons.OKCancel) != DialogResult.OK) return 1602;
        string dir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        try {
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, "MahjongHelper.msi");
            using (Stream input = Assembly.GetExecutingAssembly().GetManifestResourceStream("payload"))
            using (FileStream file = File.Create(path)) input.CopyTo(file);
            var info = new ProcessStartInfo(Path.Combine(Environment.SystemDirectory, "msiexec.exe"),
                                            "/i \"" + path + "\" /passive");
            using (Process process = Process.Start(info)) {
                process.WaitForExit();
                int code = process.ExitCode;
                MessageBox.Show(code == 0 ? "スタートメニューから Mahjong Helper を起動できます。" :
                    "Windows Installer の終了コード: " + code, "Mahjong Helper");
                return code;
            }
        } catch (Exception ex) {
            MessageBox.Show(ex.Message, "Mahjong Helper"); return 1;
        } finally {
            string path = Path.Combine(dir, "MahjongHelper.msi");
            if (File.Exists(path)) File.Delete(path);
            if (Directory.Exists(dir)) Directory.Delete(dir);
        }
    }
}
