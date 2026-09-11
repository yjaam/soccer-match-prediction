# app.R — Weekly Football Predictions Dashboard (read-only, with team logos)
#
# Reads from weekly_results/:
#   - weekly_predictions_latest.csv          (always the newest run)
#   - weekly_predictions_YYYY-MM-DD.csv      (append-only history, one per run)
#
# Team logos:
#   - Source: misc/logos/<League Folder>/<Team>.png, mounted at /logos
#   - Identity: uses the same team-name mapping as the Python pipeline
#     (src/team_name_mapping_FINAL.py), ported to R below as TEAM_MAP.
#
# Run with:  shiny::runApp("app.R", port = 3838, launch.browser = TRUE)
# Requires:  shiny, DT, dplyr, readr, lubridate, shinyjs, jsonlite, htmltools

library(shiny)
library(DT)
library(dplyr)
library(readr)
library(lubridate)
library(shinyjs)
library(jsonlite)
library(htmltools)


# ============================================================================
# PORTABLE PATH RESOLUTION
# ============================================================================

resolve_project_root <- function() {
  env_root <- Sys.getenv("PROJECT_ROOT", unset = "")
  if (nzchar(env_root) && dir.exists(env_root)) {
    return(normalizePath(env_root, mustWork = TRUE))
  }

  cwd <- getwd()
  if (dir.exists(file.path(cwd, "5_new_prediction_dashboard"))) {
    return(normalizePath(cwd, mustWork = TRUE))
  }

  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)
    candidate <- dirname(script_path)
    if (dir.exists(file.path(candidate, "5_new_prediction_dashboard"))) {
      return(normalizePath(candidate, mustWork = TRUE))
    }
  }

  stop("Could not resolve PROJECT_ROOT. Set the PROJECT_ROOT env var or run from the project root.")
}

PROJECT_ROOT <- resolve_project_root()
PIPELINE_DIR <- file.path(PROJECT_ROOT, "5_new_prediction_dashboard")
WEEKLY_RESULTS_DIR <- file.path(PIPELINE_DIR, "weekly_results")
LATEST_PRED_PATH <- file.path(WEEKLY_RESULTS_DIR, "weekly_predictions_latest.csv")
SUMMARY_PATH <- file.path(WEEKLY_RESULTS_DIR, "lstm_summary.json")
LOGOS_DIR <- file.path(PROJECT_ROOT, "misc", "logos")

message("PROJECT_ROOT:   ", PROJECT_ROOT)
message("WEEKLY_RESULTS: ", WEEKLY_RESULTS_DIR)
message("LATEST_PRED:    ", LATEST_PRED_PATH)
message("LOGOS_DIR:      ", LOGOS_DIR)


# ============================================================================
# TEAM NAME MAPPING (ported from src/team_name_mapping_FINAL.py)
# ============================================================================
#
# Every name variant → canonical 3-letter code. Both the CSV-side team names
# AND the logo-side basenames are resolved through this map, so identity is
# exact: two different teams can never share a code, and every variant of the
# same team shares one.

TEAM_MAP <- local({
  m <- c(
    # ==========================================================================
    # ORIGINAL LIST — Full official club names
    # ==========================================================================
    'FC Girondins Bordeaux' = 'BOR',
    'Football Club de Nantes' = 'NAN',
    'Olympique de Marseille' = 'OM',
    'Football Club de Metz' = 'MET',
    'Football Club Lorient-Bretagne Sud' = 'LOR',
    'Toulouse Football Club' = 'TFC',
    'Paris Saint-Germain Football Club' = 'PSG',
    'Association sportive de Monaco Football Club' = 'ASM',
    'Lille Olympique Sporting Club' = 'LOS',
    'Racing Club de Lens' = 'RCL',
    'Stade Rennais Football Club' = 'REN',
    'Stade Reims' = 'REI',
    'SM Caen' = 'CAE',
    'Thonon Évian Grand Genève FC' = 'EVI',
    'Olympique Lyonnais' = 'OL',
    'Montpellier HSC' = 'MHSC',
    'EA Guingamp' = 'GUI',
    'SC Bastia' = 'BAS',
    "Olympique Gymnaste Club Nice Côte d'Azur" = 'NIC',
    'AS Saint-Étienne' = 'STE',
    'Elche Club de Fútbol S.A.D.' = 'ELC',
    '1. Fußball- und Sportverein Mainz 05' = 'M05',
    'Hull City' = 'HUL',
    'Levante Unión Deportiva S.A.D.' = 'LEV',
    'Athletic Club Bilbao' = 'ATH',
    'Villarreal Club de Fútbol S.A.D.' = 'VIL',
    'Club Atlético de Madrid S.A.D.' = 'ATM',
    'Liverpool Football Club' = 'LIV',
    'Arsenal Football Club' = 'ARS',
    'Crystal Palace Football Club' = 'CRY',
    'Southampton FC' = 'SOU',
    'Verein für Leibesübungen Wolfsburg' = 'WOB',
    'FC Schalke 04' = 'S04',
    'Sport-Club Freiburg' = 'SCF',
    'Verein für Bewegungsspiele Stuttgart 1893' = 'VFB',
    '1. Fußball-Club Köln' = 'KOE',
    'Sunderland Association Football Club' = 'SUN',
    'Futbol Club Barcelona' = 'FCB',
    'Chelsea Football Club' = 'CHE',
    'Manchester United Football Club' = 'MUN',
    'Deportivo de La Coruña' = 'DEP',
    'Real Sociedad de Fútbol S.A.D.' = 'RSO',
    'Córdoba CF' = 'COR',
    'Hamburger Sport Verein' = 'HSV',
    'Getafe Club de Fútbol S. A. D. Team Dubai' = 'GET',
    'Stoke City' = 'STK',
    'Leicester City' = 'LEI',
    'Swansea City' = 'SWA',
    'Aston Villa Football Club' = 'AVL',
    'Associazione Calcio Milan' = 'ACM',
    'Verona Hellas Football Club' = 'VER',
    'Eintracht Frankfurt Fußball AG' = 'SGE',
    'UD Almería' = 'ALM',
    'Valencia Club de Fútbol S. A. D.' = 'VAL',
    'Sportverein Werder Bremen von 1899' = 'SVW',
    'Rayo Vallecano de Madrid S. A. D.' = 'RAY',
    'Borussia Dortmund' = 'BVB',
    'Bayer 04 Leverkusen Fußball' = 'B04',
    'Turn- und Sportgemeinschaft 1899 Hoffenheim Fußball-Spielbetriebs' = 'TSG',
    'FC Bayern München' = 'FCB',
    'Udinese Calcio' = 'UDI',
    'Associazione Calcio Fiorentina' = 'FIO',
    'FC Empoli' = 'EMP',
    'Borussia Verein für Leibesübungen 1900 Mönchengladbach' = 'BMG',
    'West Ham United Football Club' = 'WHU',
    'Tottenham Hotspur Football Club' = 'TOT',
    'Real Club Celta de Vigo S. A. D.' = 'CEL',
    'Sevilla Fútbol Club S.A.D.' = 'SEV',
    'Reial Club Deportiu Espanyol de Barcelona S.A.D.' = 'ESP',
    'Juventus Football Club' = 'JUV',
    'Società Sportiva Lazio S.p.A.' = 'LAZ',
    'UC Sampdoria' = 'SAM',
    'Società Sportiva Calcio Napoli' = 'NAP',
    'Parma Calcio 1913' = 'PAR',
    'Real Madrid Club de Fútbol' = 'RMA',
    'Granada CF' = 'GCF',
    'Manchester City Football Club' = 'MCI',
    'Associazione Sportiva Roma' = 'ROM',
    'Burnley Football Club' = 'BUR',
    'Newcastle United Football Club' = 'NEW',
    'Everton Football Club' = 'EVE',
    'Unione Sportiva Sassuolo Calcio' = 'SAS',
    'Hannover 96' = 'H96',
    'Queens Park Rangers' = 'QPR',
    'Football Club Internazionale Milano S.p.A.' = 'INT',
    'Palermo FC' = 'PAL',
    'Torino Calcio' = 'TOR',
    'Cagliari Calcio' = 'CAG',
    'Atalanta Bergamasca Calcio S.p.a.' = 'ATA',
    'Genoa Cricket and Football Club' = 'GEN',
    'Fußball-Club Augsburg 1907' = 'FCA',
    'Cesena FC' = 'CES',
    'Málaga CF' = 'MAL',
    'Chievo Verona' = 'CHI',
    'West Bromwich Albion' = 'WBA',
    'SC Paderborn 07' = 'SCP',
    'SD Eibar' = 'EIB',
    'Association Football Club Bournemouth' = 'BOU',
    "Angers Sporting Club de l'Ouest" = 'ANG',
    'SV Darmstadt 98' = 'D98',
    'FC Ingolstadt 04' = 'ING',
    'ESTAC Troyes' = 'TRO',
    'Watford FC' = 'WAT',
    'Bologna Football Club 1909' = 'BOL',
    'Sporting Gijón' = 'GIJ',
    'Frosinone Calcio' = 'FRO',
    'Norwich City' = 'NOR',
    'AC Carpi' = 'CAR',
    'UD Las Palmas' = 'LPA',
    'Real Betis Balompié S.A.D.' = 'BET',
    'Middlesbrough FC' = 'MID',
    'CD Leganés' = 'LEG',
    'AS Nancy-Lorraine' = 'NAN',
    'Deportivo Alavés S. A. D.' = 'ALA',
    'RasenBallsport Leipzig' = 'RBL',
    'Delfino Pescara 1936' = 'PES',
    'Club Atlético Osasuna' = 'OSA',
    'FC Crotone' = 'CRO',
    'Dijon FCO' = 'DIJ',
    'Amiens SC' = 'AMI',
    'Girona Fútbol Club S. A. D.' = 'GIR',
    'Racing Club de Strasbourg Alsace' = 'RCS',
    'Brighton and Hove Albion Football Club' = 'BHA',
    'Huddersfield Town' = 'HUD',
    'Benevento Calcio' = 'BEN',
    'SPAL' = 'SPA',
    'Cardiff City' = 'CAR',
    'Fulham Football Club' = 'FUL',
    'Nîmes Olympique' = 'NIM',
    'Wolverhampton Wanderers Football Club' = 'WOL',
    'SD Huesca' = 'HUE',
    'Real Valladolid CF' = 'VLD',
    'Fortuna Düsseldorf' = 'F95',
    '1. Fußballclub Union Berlin' = 'UNB',
    'Stade brestois 29' = 'SB29',
    'Real Club Deportivo Mallorca S.A.D.' = 'MLL',
    'Sheffield United' = 'SHU',
    'Unione Sportiva Lecce' = 'LEC',
    'Brescia Calcio' = 'BRE',
    'Cádiz CF' = 'CAD',
    'Arminia Bielefeld' = 'DSC',
    'Spezia Calcio' = 'SPE',
    'Leeds United Association Football Club' = 'LEE',
    'Clermont Foot 63' = 'CLE',
    'SpVgg Greuther Fürth' = 'SGF',
    'Brentford Football Club' = 'BRE',
    'VfL Bochum' = 'BOC',
    'Venezia FC' = 'VEN',
    'US Salernitana 1919' = 'SAL',
    'Nottingham Forest Football Club' = 'NFO',
    'Association de la Jeunesse auxerroise' = 'AJA',
    'AC Ajaccio' = 'ACA',
    'Unione Sportiva Cremonese S.p.A.' = 'CRE',
    'AC Monza' = 'MON',
    'Le Havre Athletic Club' = 'HAC',
    '1. Fußballclub Heidenheim 1846' = 'HDH',
    'Luton Town' = 'LUT',
    'Ipswich Town' = 'IPS',
    'Calcio Como' = 'COM',
    'Fußball-Club St. Pauli von 1910' = 'STP',
    'Holstein Kiel' = 'KSV',
    'Turn- und Sportverein Egger Glas Hartberg' = 'TSV',
    'Fußballclub Red Bull Salzburg' = 'RBS',
    'Fußballclub Blau-Weiß Linz' = 'BWL',
    'Real Oviedo S.A.D.' = 'OVI',
    'Paris Football Club' = 'PFC',
    'Pisa Sporting Club' = 'PIS',

    # ==========================================================================
    # SECOND LIST — Short / common names
    # ==========================================================================
    'Arsenal' = 'ARS',
    'Leicester' = 'LEI',
    'Man United' = 'MUN',
    'QPR' = 'QPR',
    'Stoke' = 'STK',
    'West Brom' = 'WBA',
    'West Ham' = 'WHU',
    'Liverpool' = 'LIV',
    'Newcastle' = 'NEW',
    'Burnley' = 'BUR',
    'Aston Villa' = 'AVL',
    'Chelsea' = 'CHE',
    'Crystal Palace' = 'CRY',
    'Everton' = 'EVE',
    'Southampton' = 'SOU',
    'Swansea' = 'SWA',
    'Hull' = 'HUL',
    'Sunderland' = 'SUN',
    'Tottenham' = 'TOT',
    'Man City' = 'MCI',
    'Bournemouth' = 'BOU',
    'Norwich' = 'NOR',
    'Watford' = 'WAT',
    'Middlesbrough' = 'MID',
    'Brighton' = 'BHA',
    'Huddersfield' = 'HUD',
    'Fulham' = 'FUL',
    'Wolves' = 'WOL',
    'Cardiff' = 'CAR',
    'Sheffield United' = 'SHU',
    'Leeds' = 'LEE',
    'Brentford' = 'BRE',
    "Nott'm Forest" = 'NFO',
    'Luton' = 'LUT',
    'Ipswich' = 'IPS',
    'Bayern Munich' = 'FCB',
    'Dortmund' = 'BVB',
    'Ein Frankfurt' = 'SGE',
    'FC Koln' = 'KOE',
    'Hannover' = 'H96',
    'Hertha' = 'BSC',
    'Hoffenheim' = 'TSG',
    "M'gladbach" = 'BMG',
    'Paderborn' = 'SCP',
    'Augsburg' = 'FCA',
    'Hamburg' = 'HSV',
    'Leverkusen' = 'B04',
    'Schalke 04' = 'S04',
    'Stuttgart' = 'VFB',
    'Werder Bremen' = 'SVW',
    'Wolfsburg' = 'WOB',
    'Freiburg' = 'SCF',
    'Mainz' = 'M05',
    'Darmstadt' = 'D98',
    'Ingolstadt' = 'ING',
    'RB Leipzig' = 'RBL',
    'Fortuna Dusseldorf' = 'F95',
    'Nurnberg' = 'FCN',
    'Union Berlin' = 'UNB',
    'Bielefeld' = 'DSC',
    'Bochum' = 'BOC',
    'Greuther Furth' = 'SGF',
    'Heidenheim' = 'HDH',
    'St Pauli' = 'STP',
    'Holstein Kiel' = 'KSV',
    'Chievo' = 'CHI',
    'Roma' = 'ROM',
    'Atalanta' = 'ATA',
    'Cesena' = 'CES',
    'Genoa' = 'GEN',
    'Milan' = 'ACM',
    'Palermo' = 'PAL',
    'Sassuolo' = 'SAS',
    'Torino' = 'TOR',
    'Udinese' = 'UDI',
    'Empoli' = 'EMP',
    'Juventus' = 'JUV',
    'Cagliari' = 'CAG',
    'Fiorentina' = 'FIO',
    'Inter' = 'INT',
    'Lazio' = 'LAZ',
    'Napoli' = 'NAP',
    'Parma' = 'PAR',
    'Sampdoria' = 'SAM',
    'Verona' = 'VER',
    'Frosinone' = 'FRO',
    'Bologna' = 'BOL',
    'Carpi' = 'CAR',
    'Pescara' = 'PES',
    'Crotone' = 'CRO',
    'Benevento' = 'BEN',
    'Spal' = 'SPA',
    'Lecce' = 'LEC',
    'Brescia' = 'BRE',
    'Spezia' = 'SPE',
    'Salernitana' = 'SAL',
    'Venezia' = 'VEN',
    'Monza' = 'MON',
    'Cremonese' = 'CRE',
    'Como' = 'COM',
    'Pisa' = 'PIS',
    'Almeria' = 'ALM',
    'Granada' = 'GCF',
    'Malaga' = 'MAL',
    'Sevilla' = 'SEV',
    'Barcelona' = 'FCB',
    'Celta' = 'CEL',
    'Eibar' = 'EIB',
    'Levante' = 'LEV',
    'Real Madrid' = 'RMA',
    'Vallecano' = 'RAY',
    'Getafe' = 'GET',
    'Valencia' = 'VAL',
    'Ath Bilbao' = 'ATH',
    'Ath Madrid' = 'ATM',
    'Cordoba' = 'COR',
    'Espanol' = 'ESP',
    'Elche' = 'ELC',
    'La Coruna' = 'DEP',
    'Sociedad' = 'RSO',
    'Villarreal' = 'VIL',
    'Betis' = 'BET',
    'Sp Gijon' = 'GIJ',
    'Las Palmas' = 'LPA',
    'Leganes' = 'LEG',
    'Osasuna' = 'OSA',
    'Alaves' = 'ALA',
    'Girona' = 'GIR',
    'Valladolid' = 'VLD',
    'Huesca' = 'HUE',
    'Mallorca' = 'MLL',
    'Cadiz' = 'CAD',
    'Oviedo' = 'OVI',
    'Reims' = 'REI',
    'Bastia' = 'BAS',
    'Evian Thonon Gaillard' = 'EVI',
    'Guingamp' = 'GUI',
    'Lille' = 'LOS',
    'Montpellier' = 'MHSC',
    'Nantes' = 'NAN',
    'Nice' = 'NIC',
    'Lyon' = 'OL',
    'Monaco' = 'ASM',
    'Caen' = 'CAE',
    'Lens' = 'RCL',
    'Lorient' = 'LOR',
    'Metz' = 'MET',
    'Paris SG' = 'PSG',
    'Rennes' = 'REN',
    'Toulouse' = 'TFC',
    'Bordeaux' = 'BOR',
    'Marseille' = 'OM',
    'St Etienne' = 'STE',
    'Troyes' = 'TRO',
    'Angers' = 'ANG',
    'Ajaccio GFCO' = 'GFCA',
    'Dijon' = 'DIJ',
    'Nancy' = 'NAN',
    'Amiens' = 'AMI',
    'Strasbourg' = 'RCS',
    'Nimes' = 'NIM',
    'Brest' = 'SB29',
    'Clermont' = 'CLE',
    'Ajaccio' = 'ACA',
    'Auxerre' = 'AJA',
    'Le Havre' = 'HAC',
    'Paris FC' = 'PFC',

    # ==========================================================================
    # THIRD LIST — xG data naming conventions
    # ==========================================================================
    'Paris Saint Germain' = 'PSG',
    'Saint-Etienne' = 'STE',
    'Manchester United' = 'MUN',
    'Queens Park Rangers' = 'QPR',
    'West Bromwich Albion' = 'WBA',
    'Newcastle United' = 'NEW',
    'Eintracht Frankfurt' = 'SGE',
    'FC Cologne' = 'KOE',
    'Hannover 96' = 'H96',
    'Hertha Berlin' = 'BSC',
    'Borussia M.Gladbach' = 'BMG',
    'Celta Vigo' = 'CEL',
    'Bayer Leverkusen' = 'B04',
    'Hamburger SV' = 'HSV',
    'Schalke 04' = 'S04',
    'VfB Stuttgart' = 'VFB',
    'Athletic Club' = 'ATH',
    'Atletico Madrid' = 'ATM',
    'Espanyol' = 'ESP',
    'Mainz 05' = 'M05',
    'Deportivo La Coruna' = 'DEP',
    'Real Sociedad' = 'RSO',
    'AC Milan' = 'ACM',
    'GFC Ajaccio' = 'GFCA',
    'Sporting Gijon' = 'GIJ',
    'Real Betis' = 'BET',
    'RasenBallsport Leipzig' = 'RBL',
    'SPAL 2013' = 'SPA',
    'Wolverhampton Wanderers' = 'WOL',
    'Fortuna Duesseldorf' = 'F95',
    'Real Valladolid' = 'VLD',
    'Nuernberg' = 'FCN',
    'Clermont Foot' = 'CLE',
    'Greuther Fuerth' = 'SGF',
    'FC Heidenheim' = 'HDH',
    'St. Pauli' = 'STP',

    # ==========================================================================
    # FOURTH LIST — Additional variants from match stats and lineup feeds
    # ==========================================================================
    'PSG' = 'PSG',
    'Manchester Utd' = 'MUN',
    'Nottingham' = 'NFO',
    'Frankfurt' = 'SGE',
    'Gladbach' = 'BMG',
    'Hertha BSC' = 'BSC',
    'Arminia' = 'DSC',
    'Greuther Fürth' = 'SGF',
    'Darmstadt 98' = 'D98',
    'Paderborn 07' = 'SCP',
    'Köln' = 'KOE',
    'Düsseldorf' = 'F95',
    'Nürnberg' = 'FCN',
    'Hellas Verona' = 'VER',
    'Nîmes' = 'NIM',
    'Saint-Étienne' = 'STE',
    'Evian' = 'EVI',
    'Gazélec Ajaccio' = 'GFCA',
    'Alavés' = 'ALA',
    'Almería' = 'ALM',
    'Málaga' = 'MAL',
    'Cádiz' = 'CAD',
    'Córdoba' = 'COR',
    'Leganés' = 'LEG',
    'Dep. La Coruña' = 'DEP',
    'Rayo Vallecano' = 'RAY',
    'Atlético Madrid' = 'ATM',
    'Elversberg' = 'ELV',
    'Karlsruher' = 'KSC',
    'BTSV' = 'BTS',
    '1. FC Köln' = 'KOE',
    '1899 Hoffenheim' = 'TSG',
    'FSV Mainz 05' = 'M05',
    'Mönchengladbach' = 'BMG',
    'SV 07 Elversberg' = 'ELV',
    'Athletic' = 'ATH',
    'Deportivo' = 'DEP',
    'Racing Santander' = 'RSA',
    'AJ Auxerre' = 'AJA',
    'Le Mans' = 'LEM',
    'Paris Saint-Germain' = 'PSG',
    'AFC Bournemouth' = 'BOU',
    'Brighton & Hove Albion' = 'BHA',
    'Coventry City' = 'COV',
    'Tottenham Hotspur' = 'TOT',
    'Inter Milan' = 'INT'
  )
  m
})

# All known codes — used for the "is this already a code?" fast path.
ALL_TEAM_CODES <- unique(unname(TEAM_MAP))

# Port of src/team_name_mapping_FINAL.py::resolve_team_name_fuzzy
resolve_team_code <- function(name) {
  if (is.null(name) || length(name) == 0) return(NA_character_)
  name <- as.character(name)
  if (is.na(name) || !nzchar(name)) return(NA_character_)
  name <- trimws(name)

  # Fast path: already a code?
  if (toupper(name) %in% ALL_TEAM_CODES) return(toupper(name))

  # 1. Direct lookup
  hit <- unname(TEAM_MAP[name])
  if (length(hit) == 1 && !is.na(hit)) return(hit)

  # 2. Case-insensitive
  idx <- which(tolower(names(TEAM_MAP)) == tolower(name))
  if (length(idx) > 0) return(unname(TEAM_MAP[idx[1]]))

  # 3. Suffix stripping
  for (suffix in c(" FC", " CF", " SC", " S.A.D.", " S. A. D.", " Football Club")) {
    if (endsWith(name, suffix)) {
      short <- trimws(sub(suffix, "", name, fixed = TRUE))
      hit <- unname(TEAM_MAP[short])
      if (length(hit) == 1 && !is.na(hit)) return(hit)
    }
  }

  # 4. Prefix stripping
  for (prefix in c("FC ", "AC ", "AS ", "SD ", "CD ", "SC ", "UD ", "1. ")) {
    if (startsWith(name, prefix)) {
      short <- trimws(sub(prefix, "", name, fixed = TRUE))
      hit <- unname(TEAM_MAP[short])
      if (length(hit) == 1 && !is.na(hit)) return(hit)
    }
  }

  NA_character_
}

# Accent-insensitive + punctuation-insensitive fallback normalizer.
# Used only when resolve_team_code() fails on BOTH the CSV name and the
# logo basename — that's the rare "unknown team" path.
normalize_team_name <- function(x) {
  if (is.null(x) || length(x) == 0) return("")
  x <- as.character(x)
  if (is.na(x) || !nzchar(x)) return("")
  x <- suppressWarnings(iconv(x, from = "", to = "ASCII//TRANSLIT"))
  if (is.na(x)) return("")
  x <- tolower(x)
  x <- gsub("^\\s*[0-9]+\\.\\s*", "", x)
  x <- gsub("\\b(fc|afc|cf|sc|sv|vfb|vfl|ac|as|ss|ssc|us|rc|ogc|losc|gd|cd|cs|sl|cp|fk|if|bk|ff|sk|kv|kaa|krc|rsc|bsc|tsg|rb|club|calcio|de|futbol|football|the)\\b", " ", x)
  tokens <- unlist(strsplit(x, "[^a-z0-9]+", perl = TRUE))
  tokens <- tokens[nzchar(tokens)]
  tokens <- tokens[!grepl("^[0-9]{1,4}$", tokens)]  # drop pure-numeric tokens
  paste0(tokens, collapse = "")
}


# ============================================================================
# LOGO INDEX (code-keyed, with normalized-name fallback)
# ============================================================================

if (dir.exists(LOGOS_DIR)) {
  addResourcePath("logos", LOGOS_DIR)
}

# Map our five league display names → the folder names under misc/logos
LEAGUE_TO_LOGO_FOLDER <- c(
  "Premier League" = "England - Premier League",
  "La Liga"        = "Spain - LaLiga",
  "Serie A"        = "Italy - Serie A",
  "Bundesliga"     = "Germany - Bundesliga",
  "Ligue 1"        = "France - Ligue 1"
)

build_logo_url <- function(league_folder, base) {
  paste0("logos/", utils::URLencode(league_folder, reserved = TRUE),
         "/", utils::URLencode(paste0(base, ".png"), reserved = TRUE))
}

# Returns list(code_index, name_index):
#   code_index[[CODE]] = list(league_folder, base)
#   name_index[[normalized_name]] = list(league_folder, base)
build_logo_index <- function() {
  idx_code <- list()
  idx_name <- list()
  if (!dir.exists(LOGOS_DIR)) {
    return(list(code_index = idx_code, name_index = idx_name))
  }
  files <- list.files(LOGOS_DIR, pattern = "\\.png$",
                      recursive = TRUE, full.names = TRUE,
                      ignore.case = TRUE)
  for (f in files) {
    base <- sub("\\.png$", "", basename(f), ignore.case = TRUE)
    league_folder <- basename(dirname(f))
    entry <- list(league_folder = league_folder, base = base)

    code <- resolve_team_code(base)
    if (!is.na(code)) {
      if (is.null(idx_code[[code]])) idx_code[[code]] <- entry
    }

    nm <- normalize_team_name(base)
    if (nzchar(nm)) {
      if (is.null(idx_name[[nm]])) idx_name[[nm]] <- entry
    }
  }
  list(code_index = idx_code, name_index = idx_name)
}

LOGO_INDEX <- build_logo_index()
message("Indexed ", length(LOGO_INDEX$code_index), " logos by team code and ",
        length(LOGO_INDEX$name_index), " by normalized name.")

# Resolve a logo for a given (team_name, league_display).
resolve_logo_url <- function(team_name, league_display) {
  if (is.null(team_name) || length(team_name) == 0) return(NULL)
  if (is.na(team_name)) return(NULL)
  team_name <- as.character(team_name)
  if (!nzchar(team_name)) return(NULL)

  # Expected league folder (NA when unmapped)
  league_folder <- NA_character_
  if (!is.null(league_display) && length(league_display) == 1 &&
      !is.na(league_display)) {
    mapped <- unname(LEAGUE_TO_LOGO_FOLDER[as.character(league_display)])
    if (length(mapped) == 1 && !is.na(mapped)) league_folder <- mapped
  }

  # ---- Path A: exact team-code lookup via the shared mapping ----
  code <- resolve_team_code(team_name)
  if (!is.na(code)) {
    # Prefer a hit in the expected league folder, if any.
    if (!is.na(league_folder)) {
      folder_path <- file.path(LOGOS_DIR, league_folder)
      if (dir.exists(folder_path)) {
        files <- list.files(folder_path, pattern = "\\.png$",
                            full.names = TRUE, ignore.case = TRUE)
        for (f in files) {
          b <- sub("\\.png$", "", basename(f), ignore.case = TRUE)
          c2 <- resolve_team_code(b)
          if (!is.na(c2) && c2 == code) {
            return(build_logo_url(league_folder, b))
          }
        }
      }
    }
    # Fall back to the first hit anywhere.
    hit <- LOGO_INDEX$code_index[[code]]
    if (!is.null(hit)) return(build_logo_url(hit$league_folder, hit$base))
  }

  # ---- Path B: normalized-name fallback (only reached when code lookup fails) ----
  nm <- normalize_team_name(team_name)
  if (!nzchar(nm)) return(NULL)

  # B1. Exact normalized-name match
  hit <- LOGO_INDEX$name_index[[nm]]
  if (!is.null(hit)) {
    if (!is.na(league_folder) && !identical(hit$league_folder, league_folder)) {
      folder_path <- file.path(LOGOS_DIR, league_folder)
      if (dir.exists(folder_path)) {
        files <- list.files(folder_path, pattern = "\\.png$",
                            full.names = TRUE, ignore.case = TRUE)
        for (f in files) {
          b <- sub("\\.png$", "", basename(f), ignore.case = TRUE)
          if (identical(normalize_team_name(b), nm)) {
            return(build_logo_url(league_folder, b))
          }
        }
      }
    }
    return(build_logo_url(hit$league_folder, hit$base))
  }

  # B2. Substring match — very tightly constrained (key >= 4 chars,
  # ≥ 55% overlap, longest-overlap wins). This only fires for teams
  # missing from TEAM_MAP, which should be rare.
  if (nchar(nm) < 4) return(NULL)
  folders <- if (!is.na(league_folder) && dir.exists(file.path(LOGOS_DIR, league_folder))) {
    league_folder
  } else {
    unique(vapply(LOGO_INDEX$name_index, function(h) h$league_folder, character(1)))
  }

  best <- NULL
  for (lf in folders) {
    folder_path <- file.path(LOGOS_DIR, lf)
    if (!dir.exists(folder_path)) next
    files <- list.files(folder_path, pattern = "\\.png$",
                        full.names = TRUE, ignore.case = TRUE)
    for (f in files) {
      b <- sub("\\.png$", "", basename(f), ignore.case = TRUE)
      k2 <- normalize_team_name(b)
      if (nchar(k2) < 4) next
      shorter <- if (nchar(nm) <= nchar(k2)) nm else k2
      longer  <- if (nchar(nm) <= nchar(k2)) k2 else nm
      if (!grepl(shorter, longer, fixed = TRUE)) next
      overlap <- nchar(shorter) / nchar(longer)
      if (overlap < 0.55) next
      score <- nchar(shorter)
      if (is.null(best) || score > best$score ||
          (score == best$score && nchar(b) < nchar(best$base))) {
        best <- list(score = score, league_folder = lf, base = b)
      }
    }
    if (!is.null(best) && !is.na(league_folder) && identical(lf, league_folder)) break
  }
  if (!is.null(best)) return(build_logo_url(best$league_folder, best$base))
  NULL
}

# Return an <img> tag (or "") for a team name + league, fully defensive.
team_logo_img <- function(team_name, league_display, size = 20) {
  if (is.null(team_name) || length(team_name) == 0) return("")
  if (is.na(team_name)) return("")
  if (!nzchar(as.character(team_name))) return("")

  url <- tryCatch(resolve_logo_url(team_name, league_display),
                  error = function(e) NULL)
  if (is.null(url)) return("")

  as.character(htmltools::img(
    src = url,
    style = sprintf(
      "height:%dpx;width:%dpx;object-fit:contain;vertical-align:middle;margin-right:6px;",
      size, size),
    alt = as.character(team_name)
  ))
}


# ============================================================================
# DATA HELPERS
# ============================================================================

LEAGUE_ORDER <- c("Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1")

REQUIRED_COLS <- c(
  "game_id", "league", "home_team", "away_team",
  "predicted_home_goals", "predicted_away_goals", "predicted_goal_diff",
  "prob_home_win", "prob_draw", "prob_away_win",
  "predicted_outcome", "confidence"
)

discover_dated_prediction_files <- function() {
  if (!dir.exists(WEEKLY_RESULTS_DIR)) {
    return(data.frame(date = as.Date(character()), path = character(),
                      label = character(), stringsAsFactors = FALSE))
  }
  files <- list.files(WEEKLY_RESULTS_DIR,
                      pattern = "^weekly_predictions_[0-9]{4}-[0-9]{2}-[0-9]{2}\\.csv$",
                      full.names = TRUE)
  if (length(files) == 0) {
    return(data.frame(date = as.Date(character()), path = character(),
                      label = character(), stringsAsFactors = FALSE))
  }
  dates <- as.Date(sub("^weekly_predictions_([0-9-]+)\\.csv$", "\\1", basename(files)))
  df <- data.frame(date = dates, path = files, stringsAsFactors = FALSE)
  df <- df[order(df$date, decreasing = TRUE), , drop = FALSE]
  df$label <- format(df$date, "%a %d %b %Y")
  rownames(df) <- NULL
  df
}

read_prediction_csv <- function(path) {
  if (is.null(path) || !file.exists(path)) return(NULL)
  tryCatch({
    df <- readr::read_csv(path, show_col_types = FALSE)
    for (col in REQUIRED_COLS) if (!col %in% names(df)) df[[col]] <- NA

    df <- df %>%
      mutate(
        date_str = sub("_.*$", "", game_id),
        kickoff  = suppressWarnings(lubridate::ymd(date_str))
      )

    present <- unique(df$league)
    lvls <- c(intersect(LEAGUE_ORDER, present), setdiff(present, LEAGUE_ORDER))
    df$league <- factor(df$league, levels = lvls)
    df
  }, error = function(e) {
    warning("Failed to read ", path, ": ", conditionMessage(e))
    NULL
  })
}

load_summary <- function() {
  if (!file.exists(SUMMARY_PATH)) return(NULL)
  tryCatch(jsonlite::fromJSON(SUMMARY_PATH), error = function(e) NULL)
}

file_mtime <- function(path) {
  if (is.null(path) || !file.exists(path)) return(NA)
  file.info(path)$mtime
}


# ============================================================================
# UI
# ============================================================================

ui <- fluidPage(
  useShinyjs(),
  tags$head(
    tags$style(HTML("
      body { background-color: #f5f7fa; font-family: 'Inter', system-ui, sans-serif; }
      .app-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        color: white; padding: 24px 32px; border-radius: 12px;
        margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
      }
      .app-header h1 { margin: 0; font-weight: 700; font-size: 28px; }
      .app-header p { margin: 6px 0 0 0; opacity: 0.85; font-size: 14px; }
      .stat-card {
        background: white; border-radius: 10px; padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05); border-left: 4px solid #3949ab;
      }
      .stat-card .value { font-size: 32px; font-weight: 700; color: #1a237e; }
      .stat-card .label { font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
      .stat-card.updated { border-left-color: #f9a825; }
      .stat-card.updated .value { font-size: 16px; color: #455a64; }
      .well { background: white; border: none; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }
      .table-container { background: white; padding: 16px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }
      .refresh-btn {
        background-color: #f9a825; color: white; border: none;
        font-weight: 600; padding: 10px 20px; border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: all 0.2s;
        position: relative;
      }
      .refresh-btn:hover { background-color: #f57f17; color: white; transform: translateY(-1px); }
      .refresh-btn:disabled { background-color: #bdbdbd; cursor: not-allowed; }
      .refresh-btn.loading { background-color: #f57f17; }
      .refresh-btn .btn-spinner {
        display: inline-block; width: 13px; height: 13px;
        border: 2px solid rgba(255,255,255,0.4);
        border-top-color: #fff; border-radius: 50%;
        animation: spin 0.7s linear infinite;
        vertical-align: -2px; margin-right: 8px;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
      .flash { animation: flashBg 0.8s ease-out; }
      @keyframes flashBg {
        0%   { background: #fff8e1; }
        100% { background: transparent; }
      }
      #latest_date_btn { width: 100%; padding: 6px 12px; font-size: 14px; }
      #date_filter_hint { color: #666; font-size: 13px; padding-top: 8px; }
      .empty-banner {
        background: #fff3e0; border-left: 4px solid #ef6c00; color: #4e342e;
        padding: 14px 18px; border-radius: 8px; margin-bottom: 16px; font-size: 14px;
      }
      .empty-banner code { background: rgba(0,0,0,0.06); padding: 1px 5px; border-radius: 3px; }
      table.dataTable tbody td { vertical-align: middle !important; }
    "))
  ),

  div(class = "app-header",
      fluidRow(
        column(8,
               h1("⚽ Weekly Football Predictions"),
               p(textOutput("header_subtitle", inline = TRUE))),
        column(4, style = "text-align: right; padding-top: 20px;",
               uiOutput("refresh_button_ui"))
      )
  ),

  uiOutput("error_banner"),

  fluidRow(
    column(3, div(class = "stat-card",
                  div(class = "value", textOutput("n_fixtures")),
                  div(class = "label", "Upcoming Fixtures"))),
    column(3, div(class = "stat-card",
                  div(class = "value", textOutput("n_leagues")),
                  div(class = "label", "Leagues Covered"))),
    column(3, div(class = "stat-card",
                  div(class = "value", textOutput("date_range")),
                  div(class = "label", "Date Range"))),
    column(3, div(class = "stat-card updated",
                  div(class = "value", textOutput("last_updated")),
                  div(class = "label", "Data Last Updated")))
  ),

  br(),

  wellPanel(
    fluidRow(
      column(4, selectInput("league_filter", "League",
                            choices = "All", selected = "All", multiple = TRUE)),
      column(4, selectInput("outcome_filter", "Predicted Outcome",
                            choices = c("All", "HOME_WIN", "DRAW", "AWAY_WIN"),
                            selected = "All", multiple = TRUE)),
      column(4, numericInput("min_confidence", "Minimum Confidence",
                             value = 0, min = 0, max = 1, step = 0.05))
    ),
    fluidRow(
      column(5, selectInput("week_filter", "Prediction week",
                            choices = c("Latest" = "__latest__"),
                            selected = "__latest__", multiple = FALSE)),
      column(3, div(style = "padding-top: 25px;",
                    actionButton("latest_date_btn", "⏭ Most recent",
                                 class = "refresh-btn"))),
      column(4, div(style = "padding-top: 25px;",
                    textOutput("date_filter_hint", inline = TRUE)))
    ),
    fluidRow(
      column(12, checkboxInput("group_by_league",
                               "Group by league (top of table)",
                               value = TRUE))
    )
  ),

  div(class = "table-container", DTOutput("predictions_table")),

  br(),

  wellPanel(
    h4("Model Summary"),
    verbatimTextOutput("model_summary_text")
  )
)


# ============================================================================
# SERVER
# ============================================================================

server <- function(input, output, session) {

  reload_trigger <- reactiveVal(0)
  refreshing <- reactiveVal(FALSE)

  output$refresh_button_ui <- renderUI({
    if (isTRUE(refreshing())) {
      actionButton("refresh_btn", tagList(
        tags$span(class = "btn-spinner"),
        "Refreshing..."
      ), class = "refresh-btn loading", disabled = TRUE)
    } else {
      actionButton("refresh_btn", "🔄 Refresh", class = "refresh-btn")
    }
  })

  observeEvent(input$refresh_btn, {
    req(input$refresh_btn > 0)
    refreshing(TRUE)
    Sys.sleep(0.6)
    reload_trigger(reload_trigger() + 1)
    session$onFlushed(function() {
      Sys.sleep(0.4)
      refreshing(FALSE)
    }, once = TRUE)
  })

  available_weeks <- reactive({
    reload_trigger()
    discover_dated_prediction_files()
  })

  observeEvent(available_weeks(), {
    weeks <- available_weeks()
    if (nrow(weeks) == 0) {
      updateSelectInput(session, "week_filter",
                        choices = c("Latest" = "__latest__"),
                        selected = "__latest__")
      return()
    }
    choices <- c("Latest (newest run)" = "__latest__")
    for (i in seq_len(nrow(weeks))) {
      choices[[weeks$label[i]]] <- format(weeks$date[i], "%Y-%m-%d")
    }
    current <- isolate(input$week_filter)
    if (is.null(current) || !(current %in% unname(choices))) {
      current <- "__latest__"
    }
    updateSelectInput(session, "week_filter",
                      choices = choices, selected = current)
  })

  observeEvent(input$latest_date_btn, {
    if (!identical(isolate(input$week_filter), "__latest__")) {
      updateSelectInput(session, "week_filter", selected = "__latest__")
    }
  })

  selected_file <- reactive({
    sel <- input$week_filter
    if (is.null(sel) || sel == "__latest__") {
      if (file.exists(LATEST_PRED_PATH)) return(LATEST_PRED_PATH)
      weeks <- available_weeks()
      if (nrow(weeks) > 0) return(weeks$path[1])
      return(NULL)
    }
    weeks <- available_weeks()
    hit <- weeks$path[weeks$date == as.Date(sel)]
    if (length(hit) > 0) return(hit[1])
    guess <- file.path(WEEKLY_RESULTS_DIR,
                       paste0("weekly_predictions_", sel, ".csv"))
    if (file.exists(guess)) return(guess)
    NULL
  })

  predictions <- reactive({
    reload_trigger()
    path <- selected_file()
    if (is.null(path)) return(NULL)
    read_prediction_csv(path)
  })

  output$error_banner <- renderUI({
    if (!dir.exists(WEEKLY_RESULTS_DIR)) {
      return(div(class = "empty-banner",
                 HTML(sprintf(
                   "Predictions folder not found at <code>%s</code>. Run <code>update_dashboard.py</code> first.",
                   WEEKLY_RESULTS_DIR))))
    }
    weeks <- available_weeks()
    if (is.null(selected_file()) && nrow(weeks) == 0 && !file.exists(LATEST_PRED_PATH)) {
      return(div(class = "empty-banner",
                 HTML(sprintf(
                   "No prediction files found in <code>%s</code>. Expected <code>weekly_predictions_latest.csv</code> or <code>weekly_predictions_YYYY-MM-DD.csv</code>. Run <code>update_dashboard.py</code> to generate them.",
                   WEEKLY_RESULTS_DIR))))
    }
    NULL
  })

  observeEvent(predictions(), {
    df <- predictions()
    if (is.null(df) || nrow(df) == 0) {
      updateSelectInput(session, "league_filter",
                        choices = c("All"), selected = "All")
      return()
    }
    updateSelectInput(session, "league_filter",
      choices = c("All", as.character(unique(df$league))),
      selected = "All")
  })

  output$header_subtitle <- renderText({
    df <- predictions()
    if (is.null(df) || nrow(df) == 0)
      return("No predictions file found. Run update_dashboard.py to generate one.")
    paste0(nrow(df), " matches · ", length(unique(df$league)), " leagues")
  })

  output$n_fixtures <- renderText({
    df <- predictions(); if (is.null(df)) return("—"); as.character(nrow(df))
  })
  output$n_leagues <- renderText({
    df <- predictions(); if (is.null(df)) return("—"); as.character(length(unique(df$league)))
  })
  output$date_range <- renderText({
    df <- predictions()
    if (is.null(df) || all(is.na(df$kickoff))) return("—")
    rng <- range(df$kickoff, na.rm = TRUE)
    if (rng[1] == rng[2]) return(format(rng[1], "%b %d, %Y"))
    paste0(format(rng[1], "%b %d"), " – ", format(rng[2], "%b %d"))
  })
  output$last_updated <- renderText({
    mtime <- file_mtime(selected_file())
    if (is.na(mtime)) return("no data")
    paste0("Updated: ", format(mtime, "%Y-%m-%d %H:%M"))
  })

  output$date_filter_hint <- renderText({
    weeks <- available_weeks()
    n_weeks <- nrow(weeks)
    sel <- input$week_filter
    if (is.null(sel) || sel == "__latest__") {
      return(sprintf("Showing latest run · %d week%s available",
                     n_weeks, if (n_weeks == 1) "" else "s"))
    }
    sprintf("Showing run from %s · %d week%s available",
            format(as.Date(sel), "%a %d %b %Y"),
            n_weeks, if (n_weeks == 1) "" else "s")
  })

  filtered <- reactive({
    df <- predictions()
    if (is.null(df) || nrow(df) == 0) return(NULL)

    if (!is.null(input$league_filter) && !("All" %in% input$league_filter))
      df <- df %>% filter(as.character(league) %in% input$league_filter)
    if (!is.null(input$outcome_filter) && !("All" %in% input$outcome_filter))
      df <- df %>% filter(predicted_outcome %in% input$outcome_filter)
    df <- df %>% filter(is.na(confidence) | confidence >= input$min_confidence)

    if (isTRUE(input$group_by_league)) {
      df %>% arrange(league, kickoff, home_team)
    } else {
      df %>% arrange(kickoff, home_team)
    }
  })

  output$predictions_table <- renderDT({
    df <- filtered()
    if (is.null(df) || nrow(df) == 0) {
      return(datatable(
        data.frame(Message = "No predictions match the current filters. If nothing has been scraped yet, run update_dashboard.py to generate predictions."),
        options = list(dom = 't'), rownames = FALSE))
    }

    display <- df
    display$League <- as.character(display$league)
    display$Date   <- format(display$kickoff, "%a %d %b")

    logo_cache <- new.env(parent = emptyenv())
    render_team_cell <- function(team, league) {
      team_safe   <- if (is.null(team)   || length(team)   == 0 || is.na(team))   "" else as.character(team)
      league_safe <- if (is.null(league) || length(league) == 0 || is.na(league)) "" else as.character(league)

      key <- paste(team_safe, league_safe, sep = "||")
      cached <- logo_cache[[key]]
      if (!is.null(cached)) return(cached)

      img <- tryCatch(team_logo_img(team_safe, league_safe),
                      error = function(e) "")
      tag <- paste0(img, "<span>", htmltools::htmlEscape(team_safe), "</span>")
      logo_cache[[key]] <- tag
      tag
    }

    display$Home <- mapply(
      function(tm, lg) render_team_cell(tm, lg),
      as.character(display$home_team),
      as.character(display$League),
      USE.NAMES = FALSE, SIMPLIFY = TRUE
    )
    display$Away <- mapply(
      function(tm, lg) render_team_cell(tm, lg),
      as.character(display$away_team),
      as.character(display$League),
      USE.NAMES = FALSE, SIMPLIFY = TRUE
    )

    out <- data.frame(
      League     = display$League,
      Date       = display$Date,
      Home       = display$Home,
      Away       = display$Away,
      `λ Home`   = display$predicted_home_goals,
      `λ Away`   = display$predicted_away_goals,
      `Δ`        = display$predicted_goal_diff,
      `P(H)`     = display$prob_home_win,
      `P(D)`     = display$prob_draw,
      `P(A)`     = display$prob_away_win,
      Prediction = display$predicted_outcome,
      Confidence = display$confidence,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )

    datatable(
      out, rownames = FALSE, escape = FALSE,
      options = list(
        pageLength = 25,
        lengthMenu = c(10, 25, 50, 100),
        order = list(list(1, 'asc')),
        columnDefs = list(
          list(className = 'dt-center',
               targets = c(0, 1, 4, 5, 6, 7, 8, 9, 10, 11)),
          list(className = 'dt-left', targets = c(2, 3))
        ),
        dom = 'lftip'
      )
    ) %>%
      formatRound(c("λ Home", "λ Away", "Δ"), 2) %>%
      formatPercentage(c("P(H)", "P(D)", "P(A)", "Confidence"), 1) %>%
      formatStyle("Prediction",
        backgroundColor = styleEqual(
          c("HOME_WIN", "DRAW", "AWAY_WIN"),
          c("#e8f5e9", "#fff8e1", "#ffebee")),
        color = styleEqual(
          c("HOME_WIN", "DRAW", "AWAY_WIN"),
          c("#1b5e20", "#e65100", "#b71c1c")),
        fontWeight = 'bold'
      ) %>%
      formatStyle("Confidence",
        background = styleColorBar(c(0, 1), "#90caf9"),
        backgroundSize = '100% 80%',
        backgroundRepeat = 'no-repeat',
        backgroundPosition = 'center'
      )
  })

  output$model_summary_text <- renderText({
    s <- load_summary()
    if (is.null(s)) return("No model summary available.")
    paste(c(
      paste0("Training rows:        ", s$rows$train),
      paste0("Validation rows:      ", s$rows$val),
      paste0("Upcoming fixtures:    ", s$rows$upcoming),
      paste0("Draw margin:          ", round(s$validation$draw_margin, 3)),
      paste0("Validation balanced:  ", round(s$validation$balanced_accuracy, 4)),
      paste0("Blend weight (clf):   ", s$blend_weight_selected),
      paste0("Best epoch:           ", s$best_epoch)
    ), collapse = "\n")
  })
}

shinyApp(ui = ui, server = server)