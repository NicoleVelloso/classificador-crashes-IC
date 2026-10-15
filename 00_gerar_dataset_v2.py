import numpy as np
import pandas as pd

ARQUIVO_ATUAL = "Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026.xlsx"
ARQUIVO_NOVO = "Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026_v2.xlsx"

N_TREINO = 150
N_TESTE = 40

SEED = 2026
rng = np.random.default_rng(SEED)


ARQUIVOS = {
    "core": {
        "descricao": "fluxo principal do aplicativo",
        "logica": [
            ("AuthenticationService.kt", "com.example.auth",
             ["validateUserToken", "resolveIdentity", "revokeSession"]),
            ("TokenRefreshWorker.kt", "com.example.auth",
             ["refreshCredentials", "scheduleRenewal", "persistToken"]),
            ("CheckoutCoordinator.kt", "com.example.payment",
             ["commitTransaction", "confirmPayment", "finalizeOrder"]),
            ("PaymentProcessor.java", "com.example.payment",
             ["authorizeCharge", "captureAmount", "rollbackCharge"]),
            ("SubscriptionGateway.kt", "com.example.payment",
             ["renewPlan", "validateReceipt", "syncEntitlements"]),
            ("OnboardingFlowManager.kt", "com.example.onboarding",
             ["completeEnrollment", "advanceStep", "persistUserProfile"]),
            ("DeviceSyncEngine.kt", "com.example.sync",
             ["uploadPendingRecords", "mergeRemoteState", "flushQueue"]),
            ("HealthDataUploader.swift", "com.example.sync",
             ["sendSamples", "reconcileHistory", "commitBatch"]),
            ("CredentialStore.kt", "com.example.security",
             ["writeSecret", "readSecret", "rotateKey"]),
        ],
        "ui": [
            ("LoginFormView.kt", "com.example.auth",
             ["renderFields", "layoutActions", "applyErrorState"]),
            ("CheckoutSummaryView.kt", "com.example.payment",
             ["renderTotal", "layoutLineItems", "bindCurrencyLabel"]),
            ("OnboardingStepView.kt", "com.example.onboarding",
             ["renderProgress", "layoutIllustration", "bindStepTitle"]),
            ("AccountHeaderView.kt", "com.example.account",
             ["renderAvatar", "layoutBadges", "bindDisplayName"]),
            ("BiometricPromptView.swift", "com.example.security",
             ["renderPrompt", "layoutIcon", "applyTheme"]),
        ],
    },
    "secundaria": {
        "descricao": "funcionalidade secundaria",
        "logica": [
            ("FirmwareVersionChecker.kt", "com.example.device",
             ["fetchRemoteVersion", "compareBuilds", "cacheResult"]),
            ("BatteryStatusPoller.kt", "com.example.device",
             ["pollStatus", "readLevel", "scheduleNextPoll"]),
            ("StepCounterAdapter.kt", "com.example.metrics",
             ["aggregateDaily", "readSensorBuffer", "persistSeries"]),
            ("NotificationDispatcher.kt", "com.example.notify",
             ["enqueueMessage", "resolveChannel", "deliverNow"]),
            ("ReminderScheduler.swift", "com.example.notify",
             ["scheduleLocal", "cancelPending", "rebuildQueue"]),
            ("PreferencesRepository.kt", "com.example.settings",
             ["readPreference", "writePreference", "migrateSchema"]),
            ("ExportHistoryService.java", "com.example.settings",
             ["exportRange", "buildCsv", "writeToDisk"]),
        ],
        "ui": [
            ("MonthlyChartView.kt", "com.example.metrics",
             ["drawSeries", "layoutAxisLabels", "renderLegend"]),
            ("DistanceWidget.kt", "com.example.widgets",
             ["renderDistance", "layoutUnits", "applyWidgetTheme"]),
            ("WeatherPanelView.kt", "com.example.widgets",
             ["renderPanel", "layoutForecastRow", "bindTemperature"]),
            ("SettingsRowView.kt", "com.example.settings",
             ["renderRow", "layoutDivider", "bindSubtitle"]),
            ("NotificationCardView.kt", "com.example.notify",
             ["renderCard", "layoutActions", "applyAccentColor"]),
        ],
    },
    "cosmetica": {
        "descricao": "camada visual",
        "logica": [
            ("LegacyIconMapper.java", "com.example.legacy",
             ["resolveIconName", "loadLegacyAsset", "clearMapping"]),
            ("AssetCacheCleaner.kt", "com.example.ui",
             ["purgeStaleEntries", "evictOldest", "rebuildIndex"]),
        ],
        "ui": [
            ("RippleEffectAnimator.kt", "com.example.ui",
             ["drawFrame", "computeRadius", "startRipple"]),
            ("EmojiRenderer.kt", "com.example.ui",
             ["measureGlyph", "renderGlyph", "resolveCodepoint"]),
            ("TooltipAnchor.kt", "com.example.ui",
             ["computeOffset", "positionArrow", "measureBounds"]),
            ("ThemeTransition.swift", "com.example.ui",
             ["applyPalette", "interpolateColors", "commitTheme"]),
            ("AvatarPlaceholder.kt", "com.example.ui",
             ["drawInitials", "computeCornerRadius", "tintBackground"]),
            ("ShimmerLoader.kt", "com.example.ui",
             ["startShimmer", "layoutSkeleton", "fadeOut"]),
            ("DeprecatedBadgeView.java", "com.example.legacy",
             ["layoutBadge", "renderCount", "applyLegacySpacing"]),
        ],
    },
}


MODELOS_MENSAGEM = {
    "grave": [
        "{obj} discarded before the write completed",
        "{obj} written to plain log output",
        "{obj} left in an unrecoverable state",
        "{obj} lost after the rollback",
        "{obj} persisted without encryption",
    ],
    "bloqueio": [
        "{acao} cannot proceed past the current step",
        "{acao} stuck with no fallback path available",
        "{acao} never completes leaving the screen blocked",
        "{acao} aborted and the retry loop was exhausted",
    ],
    "degradacao": [
        "{obj} unavailable falling back to the cached copy",
        "{acao} timed out using the last known result",
        "{obj} returned incomplete with missing optional fields",
        "{acao} skipped the user can retry manually",
    ],
    "visual": [
        "{elem} rendered past its bounds",
        "{elem} drawn with the wrong offset",
        "{elem} missing from the active palette",
        "{elem} clipped on narrow screens",
        "{elem} laid out at the wrong scale",
    ],
}

VOCABULARIO = {
    "com.example.auth": ("session token", "sign in", "login button"),
    "com.example.payment": ("payment record", "checkout", "order total label"),
    "com.example.onboarding": ("enrollment data", "onboarding", "progress bar"),
    "com.example.sync": ("pending sync batch", "device sync", "sync status icon"),
    "com.example.security": ("stored credential", "biometric check", "prompt icon"),
    "com.example.account": ("profile record", "profile load", "account avatar"),
    "com.example.device": ("firmware status", "firmware check", "device panel row"),
    "com.example.metrics": ("step history", "metrics load", "chart axis label"),
    "com.example.widgets": ("widget state", "widget refresh", "widget tile"),
    "com.example.notify": ("reminder payload", "notification delivery", "notification card"),
    "com.example.settings": ("stored preference", "settings load", "settings row divider"),
    "com.example.ui": ("cached asset", "theme transition", "tooltip anchor"),
    "com.example.legacy": ("icon mapping", "legacy asset load", "deprecated badge"),
}


def compor_mensagem(biblioteca, dano):
    obj, acao, elem = VOCABULARIO[biblioteca]
    modelo = escolher(MODELOS_MENSAGEM[dano])
    texto = modelo.format(obj=obj, acao=acao, elem=elem)
    return texto[0].upper() + texto[1:]


DESCRICAO_DANO = {
    "grave": "perda de dados ou exposicao de informacao",
    "bloqueio": "fluxo interrompido sem alternativa",
    "degradacao": "funcionalidade degradada com alternativa",
    "visual": "defeito apenas de apresentacao",
}

CAMADAS_POR_DANO = {
    "grave": ["logica"],
    "bloqueio": ["logica", "logica", "ui"],
    "degradacao": ["logica", "ui"],
    "visual": ["ui"],
}



EXCECOES_ANDROID = {
    "grave": [
        "java.lang.SecurityException", "java.io.IOException",
        "java.lang.IllegalStateException", "java.lang.NullPointerException",
        "java.lang.RuntimeException",
    ],
    "bloqueio": [
        "java.util.concurrent.TimeoutException", "java.lang.IllegalStateException",
        "java.lang.NullPointerException", "java.lang.UnsatisfiedLinkError",
        "java.lang.RuntimeException",
    ],
    "degradacao": [
        "java.io.IOException", "java.util.concurrent.TimeoutException",
        "java.lang.NullPointerException", "java.lang.ClassCastException",
        "java.lang.NoClassDefFoundError",
    ],
    "visual": [
        "java.lang.IllegalArgumentException", "android.view.InflateException",
        "java.lang.ArrayIndexOutOfBoundsException", "java.lang.NullPointerException",
        "java.lang.IllegalStateException",
    ],
}

EXCECOES_IOS = {
    "grave": ["NSInternalInconsistencyException", "EXC_BAD_ACCESS", "NSGenericException"],
    "bloqueio": ["NSInternalInconsistencyException", "NSGenericException", "EXC_BAD_ACCESS"],
    "degradacao": ["NSGenericException", "NSInvalidArgumentException", "NSRangeException"],
    "visual": ["NSInvalidArgumentException", "NSRangeException", "NSGenericException"],
}


MATRIZ_SEVERIDADE = {
    ("core", "grave"): ("CRÍTICO", False),
    ("core", "bloqueio"): ("CRÍTICO", False),
    ("core", "degradacao"): ("MÉDIO", True),
    ("core", "visual"): ("BAIXO", True),
    ("secundaria", "grave"): ("CRÍTICO", True),
    ("secundaria", "bloqueio"): ("MÉDIO", False),
    ("secundaria", "degradacao"): ("MÉDIO", False),
    ("secundaria", "visual"): ("BAIXO", False),
    ("cosmetica", "grave"): ("MÉDIO", True),
    ("cosmetica", "bloqueio"): ("MÉDIO", True),
    ("cosmetica", "degradacao"): ("BAIXO", False),
    ("cosmetica", "visual"): ("BAIXO", False),
}


def justificar(area, dano, ambiguo):
    texto = (
        f"{ARQUIVOS[area]['descricao'].capitalize()} atingida; "
        f"{DESCRICAO_DANO[dano]}."
    )
    if ambiguo:
        texto += " Caso limitrofe: os dois criterios apontam em direcoes diferentes."
    return texto




def sortear_impacto():
    usuarios = int(np.clip(rng.lognormal(mean=4.6, sigma=1.5), 3, 25000))
    ocorrencias = int(usuarios * rng.uniform(1.05, 2.4))
    return usuarios, ocorrencias




VERSOES_APP = ["8.9.2", "9.0.4", "9.1.1", "9.2.0", "9.2.1"]
FABRICANTES_ANDROID = ["Samsung", "Xiaomi", "Motorola", "Google", "realme"]
MODELOS_ANDROID = {
    "Samsung": ["SM-A546E", "SM-S918B", "SM-A155M"],
    "Xiaomi": ["POCO F5", "Redmi Note 12"],
    "Motorola": ["Moto G84", "Edge 40"],
    "Google": ["Pixel 7a", "Pixel 8"],
    "realme": ["RMX3771"],
}
MODELOS_IOS = ["iPhone 12", "iPhone 13", "iPhone 14", "iPhone 15 Pro", "iPhone SE"]
VERSOES_ANDROID = ["12", "13", "14", "15"]
VERSOES_IOS = ["16.6", "17.2", "17.5", "18.0"]
DONOS = ["DEVELOPER", "VENDOR", "RUNTIME", "THIRD_PARTY_SDK"]


def escolher(lista):
    return lista[int(rng.integers(0, len(lista)))]
    

def gerar_crash(indice, area, dano):
    severidade, ambiguo = MATRIZ_SEVERIDADE[(area, dano)]

    camada = escolher(CAMADAS_POR_DANO[dano])
    candidatos = ARQUIVOS[area][camada] or ARQUIVOS[area]["logica"]
    arquivo, biblioteca, metodos = candidatos[int(rng.integers(0, len(candidatos)))]
    metodo = escolher(metodos)
    mensagem = compor_mensagem(biblioteca, dano)

    ios = arquivo.endswith(".swift")
    if ios:
        plataforma, so_nome = "IOS", "iOS"
        so_versao = escolher(VERSOES_IOS)
        fabricante, modelo = "Apple", escolher(MODELOS_IOS)
        excecao = escolher(EXCECOES_IOS[dano])
    else:
        plataforma, so_nome = "ANDROID", "Android"
        so_versao = escolher(VERSOES_ANDROID)
        fabricante = escolher(FABRICANTES_ANDROID)
        modelo = escolher(MODELOS_ANDROID[fabricante])
        excecao = escolher(EXCECOES_ANDROID[dano])

    usuarios, ocorrencias = sortear_impacto()

    tipo_erro = "FATAL" if rng.random() < 0.6 else "NON_FATAL"
    estado = "FOREGROUND" if rng.random() < 0.75 else "BACKGROUND"
    versao_app = escolher(VERSOES_APP)
    dia = int(rng.integers(1, 29))
    hora, minuto, seg = (
        int(rng.integers(0, 24)),
        int(rng.integers(0, 60)),
        int(rng.integers(0, 60)),
    )

    return {
        "event_id": f"evt_{indice:05d}",
        "event_timestamp": f"2026-05-{dia:02d} {hora:02d}:{minuto:02d}:{seg:02d}",
        "issue_id": f"iss_{int(rng.integers(1000, 9999))}",
        "application_display_version": versao_app,
        "application_build_version": str(
            int(versao_app.replace(".", "")) * 100 + int(rng.integers(0, 99))
        ),
        "platform": plataforma,
        "operating_system_display_version": so_versao,
        "operating_system_name": so_nome,
        "device_manufacturer": fabricante,
        "device_model": modelo,
        "error_type": tipo_erro,
        "process_state": estado,
        "exceptions_type": excecao,
        "exceptions_exception_message": mensagem,
        "blame_frame_file": arquivo,
        "blame_frame_line": int(rng.integers(12, 480)),
        "blame_frame_symbol": f"{arquivo.split('.')[0]}.{metodo}",
        "blame_frame_library": biblioteca,
        "blame_frame_owner": escolher(DONOS),
        "occurrences_count": ocorrencias,
        "users_affected": usuarios,
        "severity": severidade,
        "severity_justification": justificar(area, dano, ambiguo),
    }



DISTRIBUICAO = {"CRÍTICO": 0.24, "MÉDIO": 0.36, "BAIXO": 0.40}
FRACAO_AMBIGUA = 0.25

combinacoes_por_classe = {"CRÍTICO": [], "MÉDIO": [], "BAIXO": []}
for (area_, dano_), (sev_, ambiguo_) in MATRIZ_SEVERIDADE.items():
    combinacoes_por_classe[sev_].append((area_, dano_, ambiguo_))


def montar_conjunto(n, indice_inicial):
    registros = []
    indice = indice_inicial

    for classe, proporcao in DISTRIBUICAO.items():
        quantidade = int(round(n * proporcao))
        opcoes = combinacoes_por_classe[classe]
        ambiguas = [c for c in opcoes if c[2]]
        claras = [c for c in opcoes if not c[2]]

        for _ in range(quantidade):
            usar_ambigua = ambiguas and rng.random() < FRACAO_AMBIGUA
            fonte = ambiguas if usar_ambigua else (claras or ambiguas)
            area, dano, _ = fonte[int(rng.integers(0, len(fonte)))]
            indice += 1
            registros.append(gerar_crash(indice, area, dano))

    return registros, indice


treino, ultimo = montar_conjunto(N_TREINO, 0)
teste, _ = montar_conjunto(N_TESTE, ultimo)

df_treino = pd.DataFrame(treino).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_teste = pd.DataFrame(teste).sample(frac=1, random_state=SEED + 1).reset_index(drop=True)



print("=" * 70)
print("GERADOR DO DATASET SINTETICO")
print("=" * 70)

ABA_TREINO_PADRAO = "Crashes Treino"
ABA_TESTE_PADRAO = "Crashes Teste (Novos)"
COLUNAS_SO_DE_TREINO = ("severity", "severity_justification")

aba_treino, aba_teste = ABA_TREINO_PADRAO, ABA_TESTE_PADRAO
colunas_treino = list(df_treino.columns)
colunas_teste = [c for c in df_treino.columns if c not in COLUNAS_SO_DE_TREINO]

try:
    nomes_abas = pd.ExcelFile(ARQUIVO_ATUAL).sheet_names

    if len(nomes_abas) >= 1:
        aba_treino = nomes_abas[0]
        colunas_treino = list(
            pd.read_excel(ARQUIVO_ATUAL, sheet_name=aba_treino).columns
        )
    if len(nomes_abas) >= 2:
        aba_teste = nomes_abas[1]
        colunas_teste = list(
            pd.read_excel(ARQUIVO_ATUAL, sheet_name=aba_teste).columns
        )
    else:
        colunas_teste = [c for c in colunas_treino if c not in COLUNAS_SO_DE_TREINO]
        print(
            f"\nAviso: '{ARQUIVO_ATUAL}' tem apenas uma aba. A aba de teste sera"
            f"\ncriada com o nome padrao '{aba_teste}'."
        )

    print(f"\nFormato lido de: {ARQUIVO_ATUAL}")
    print(f"   Aba de treino: '{aba_treino}' ({len(colunas_treino)} colunas)")
    print(f"   Aba de teste:  '{aba_teste}' ({len(colunas_teste)} colunas)")

except FileNotFoundError:
    print(
        f"\nAviso: '{ARQUIVO_ATUAL}' nao encontrado nesta pasta."
        f"\nUsando os nomes de aba padrao."
    )

faltando = [c for c in colunas_treino if c not in df_treino.columns]
if faltando:
    raise SystemExit(
        f"\nERRO: o gerador nao produz as colunas {faltando}.\n"
        "Ajuste a funcao gerar_crash antes de continuar."
    )

df_treino_final = df_treino[colunas_treino]
df_teste_final = df_teste[[c for c in colunas_teste if c in df_teste.columns]]

with pd.ExcelWriter(ARQUIVO_NOVO, engine="openpyxl") as writer:
    df_treino_final.to_excel(writer, sheet_name=aba_treino, index=False)
    df_teste_final.to_excel(writer, sheet_name=aba_teste, index=False)

print(f"\nArquivo gerado: {ARQUIVO_NOVO}")
print(f"   Treino: {len(df_treino_final)} crashes rotulados")
print(f"   Teste:  {len(df_teste_final)} crashes sem rotulo")



ordem = ["CRÍTICO", "MÉDIO", "BAIXO"]
tudo_ok = True

print("\n" + "-" * 70)
print("SOBREPOSICAO DAS COLUNAS DE IMPACTO")
print("-" * 70)

for coluna in ["occurrences_count", "users_affected"]:
    print(f"\n{coluna}:")
    faixas = {}
    for classe in ordem:
        v = df_treino_final.loc[df_treino_final["severity"] == classe, coluna]
        faixas[classe] = (v.min(), v.max())
        print(
            f"   {classe:<10} min={v.min():>7.0f}  max={v.max():>7.0f}  "
            f"mediana={v.median():>7.0f}"
        )
    pares = 0
    for i in range(len(ordem)):
        for j in range(i + 1, len(ordem)):
            a, b = faixas[ordem[i]], faixas[ordem[j]]
            if a[0] <= b[1] and b[0] <= a[1]:
                pares += 1
    if pares == 3:
        print("   OK - as tres classes se sobrepoem; a coluna nao separa sozinha")
    else:
        print("   ATENCAO - ainda ha separacao entre classes nesta coluna")
        tudo_ok = False

print("\n" + "-" * 70)
print("EXCECOES COMPARTILHADAS ENTRE CLASSES")
print("-" * 70)
compartilhadas = sum(
    1
    for _, grupo in df_treino_final.groupby("exceptions_type")
    if grupo["severity"].nunique() >= 2
)
distintas = df_treino_final["exceptions_type"].nunique()
print(f"\n   {compartilhadas} de {distintas} tipos aparecem em 2 ou mais classes")

print("\n" + "-" * 70)
print("COERENCIA ENTRE PLATAFORMA E ARQUIVO")
print("-" * 70)
incoerentes = df_treino_final[
    (df_treino_final["blame_frame_file"].str.endswith(".swift"))
    & (df_treino_final["platform"] != "IOS")
]
incoerentes_android = df_treino_final[
    (~df_treino_final["blame_frame_file"].str.endswith(".swift"))
    & (df_treino_final["platform"] != "ANDROID")
]
total_incoerente = len(incoerentes) + len(incoerentes_android)
if total_incoerente == 0:
    print("\n   OK - arquivos .swift sempre em iOS, .kt e .java sempre em Android")
else:
    print(f"\n   ATENCAO - {total_incoerente} registros com plataforma incoerente")
    tudo_ok = False

print("\n" + "-" * 70)
print("DISTRIBUICAO DAS CLASSES")
print("-" * 70)
for classe in ordem:
    n = int((df_treino_final["severity"] == classe).sum())
    print(f"   {classe:<10} {n:>4} crashes ({n / len(df_treino_final) * 100:.0f}%)")

ambiguos = int(
    df_treino_final["severity_justification"].str.contains("limitrofe").sum()
)
print(f"\n   Casos limitrofes (sinais conflitantes): {ambiguos}")

print("\n" + "-" * 70)
print("AMOSTRA - UM CRASH DE CADA CLASSE")
print("-" * 70)
for classe in ordem:
    linha = df_treino_final[df_treino_final["severity"] == classe].iloc[0]
    print(f"\n{classe}")
    print(f"   Arquivo:  {linha['blame_frame_file']}")
    print(f"   Metodo:   {linha['blame_frame_symbol']}")
    print(f"   Excecao:  {linha['exceptions_type']}")
    print(f"   Mensagem: {linha['exceptions_exception_message']}")
    print(f"   Impacto:  {linha['users_affected']} usuarios")

print("\n" + "=" * 70)
if tudo_ok:
    print("Dataset gerado e verificado.")
else:
    print("Revise a geracao: alguma verificacao falhou.")
print("=" * 70)
