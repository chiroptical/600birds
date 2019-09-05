Bird sound recordings were downloaded from [Xeno-Canto](https://xeno-canto.org). The Xeno-Canto servers are not designed for bulk download; to avoid overloading the servers, I downloaded recordings only at low-traffic times, which were confirmed by contacting Xeno-Canto staff.

To reduce the chance of Xeno-Canto servers being flooded with requests, the download scripts used are not posted publicly here. However, the scripts are available on request; contact me ([Tessa Rhinehart](https://github.com/rhine3)) to obtain them.

Descriptions of other labeled datasets are available [here](http://lila.science/otherdatasets#bioacoustics).

The rest of the scripts in this repository assume that the recordings are organized in a single directory organized as follows, where `catnum`s are the catalog numbers for each recording, and the `jsons` are recording metadata obtained from Xeno-Canto.

```
downloaddate/
    |-- species-1/
    |   |-- mp3s/
    |   |   |-- catnum1.mp3
    |   |   |-- catnum2.mp3
    |   |   |-- ...
    |   |-- downloaddate-species-1.json
    |-- species-2/
    |   |-- ...
    
```
