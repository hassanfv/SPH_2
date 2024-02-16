
def selectediFrac(iF):

  iFrac = [
    # 0->HI
    iF[0, 0],
    # 1->H2
    iF[0, 2],
    # 2->CI
    iF[5, 0],
    # 3->CII
    iF[5, 1],
    # 4->CIII
    iF[5, 2],
    # 5->CIV
    iF[5, 3],
    # 6->NV
    iF[6, 4],
    # 7->OI
    iF[7, 0],
    # 8->OVI
    iF[7, 5],
    # 9->NaI
    iF[10, 0],
    # 10->MgI
    iF[11, 0],
    # 11->MgII
    iF[11, 1],
    # 12->AlII
    iF[12, 1],
    # 13->AlIII
    iF[12, 2],
    # 14->SiII
    iF[13, 1],
    # 15->SiIII
    iF[13, 2],
    # 16->SiIV
    iF[13, 3],
    # 17->PV
    iF[14, 4],
    # 18->SII
    iF[15, 1],
    # 19->CrII
    iF[23, 1],
    # 20->MnII
    iF[24, 1],
    # 21->FeII
    iF[25, 1],
    # 22->ZnII
    iF[29, 1]
  ]

  return iFrac


