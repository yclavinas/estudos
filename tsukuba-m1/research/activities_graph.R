library(DiagrammeR)
mermaid("
        gantt
        dateFormat  MM-DD-YYYY
        title MOEA/D - master research
        
        section M1
        Survey             :active,                      first_1,    04-01-2018, 10-01-2018
        Select Ideias      :crit, active,                first_2,    05-01-2018, 09-01-2018
        Implemention                 :active,            first_3,    05-15-2018, 10-01-2018
        Analyse results                       :          first_4,    10-01-2018, 12-01-2018
        
        section M2
        Improve implementation :                          first_5,    11-01-2018, 02-01-2019
        Create python wrapper :                           first_6,    02-01-2019, 02-15-2019
        Analyse results                         :         first_7,    02-15-2019, 04-15-2019
        Improve implementation       :                    first_8,    04-15-2019, 06-15-2019
        (Re)Analyse results                     :         first_9,    06-15-2019, 06-30-2019        

        section Dissertation
        Write, critical task      :crit,    import_1,   06-01-2018, 09-01-2019
        Review, critical task      :crit,    import_1,   09-01-2019, 11-15-2019
        ")
