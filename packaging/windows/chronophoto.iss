#ifndef MyAppVersion
  #define MyAppVersion "0.4.0"
#endif

[Setup]
AppId={{98B31103-786C-4BAD-B3EC-C7A2B83B6A01}
AppName=Chronophoto
AppVersion={#MyAppVersion}
AppPublisher=Chronophoto
SetupIconFile=chronophoto.ico
DefaultDirName={localappdata}\Programs\Chronophoto
DefaultGroupName=Chronophoto
DisableProgramGroupPage=yes
OutputDir=..\..\release
OutputBaseFilename=Chronophoto-Windows-x64-Setup
Compression=lzma2/max
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=Chronophoto
UninstallDisplayIcon={app}\Chronophoto.exe
WizardStyle=modern

[Files]
Source: "..\..\dist\Chronophoto\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Chronophoto"; Filename: "{app}\Chronophoto.exe"
Name: "{autodesktop}\Chronophoto"; Filename: "{app}\Chronophoto.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\Chronophoto.exe"; Description: "Launch Chronophoto"; Flags: nowait postinstall skipifsilent
